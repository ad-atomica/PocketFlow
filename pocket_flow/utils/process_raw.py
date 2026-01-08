from __future__ import annotations

from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .parse_file import Chain, Ligand, Protein
from .residues_base import RESIDUES_TOPO


def compute_dist_mat(
    m1: NDArray[np.floating[Any]], m2: NDArray[np.floating[Any]]
) -> NDArray[np.floating[Any]]:
    m1_square = np.expand_dims(np.einsum("ij,ij->i", m1, m1), axis=1)
    if m1 is m2:
        m2_square = m1_square.T
    else:
        m2_square = np.expand_dims(np.einsum("ij,ij->i", m2, m2), axis=0)
    dist_mat = m1_square + m2_square - np.dot(m1, m2.T) * 2
    # Result maybe less than 0 due to floating point rounding errors.
    dist_mat = np.maximum(dist_mat, 0)
    if m1 is m2:
        # Ensure that distances between vectors and themselves are set to 0.0.
        # This may not be the case due to floating point rounding errors.
        dist_mat.flat[:: dist_mat.shape[0] + 1] = 0.0
    return np.sqrt(dist_mat)


class SplitPocket:
    main_path: Path
    sample_path: Path
    new_sample_path: Path
    dist_cutoff: float
    type_file: Path
    types: list[list[str]]
    exceptions: list[Any]
    get_surface_atom: bool

    def __init__(
        self,
        main_path: Path | str = Path("./data/CrossDocked2020/"),
        sample_path: Path | str = Path("./Samples/"),
        new_sample_path: Path | str = Path("./Samples_Pocket/"),
        type_file: Path | str = Path("./data/CrossDocked2020/samples.types"),
        dist_cutoff: float = 10.0,
        get_surface_atom: bool = True,
    ) -> None:
        self.main_path = Path(main_path)
        self.sample_path = Path(sample_path)
        self.new_sample_path = Path(new_sample_path)
        self.dist_cutoff = dist_cutoff
        self.type_file = Path(type_file)
        with open(self.type_file) as f:
            self.types = [line.strip().split() for line in f.readlines()]
        self.exceptions = []
        self.get_surface_atom = get_surface_atom

    @staticmethod
    def _split_pocket(protein: Chain | Protein, ligand: Ligand, dist_cutoff: float) -> tuple[str, str]:
        res = np.array(protein.get_residues)
        cm_res = np.array([r.center_of_mass for r in res])
        lig_conformer = ligand.mol.GetConformer()
        lig_pos = np.array(
            [
                [
                    lig_conformer.GetAtomPosition(a.GetIdx()).x,
                    lig_conformer.GetAtomPosition(a.GetIdx()).y,
                    lig_conformer.GetAtomPosition(a.GetIdx()).z,
                ]
                for a in ligand.mol.GetAtoms()
            ]
        )
        dist_mat = compute_dist_mat(lig_pos, cm_res)
        bool_dist_mat = dist_mat < dist_cutoff
        pocket_res = res[bool_dist_mat.sum(axis=0) > 0]
        pocket_block = "\n".join(
            i.to_heavy_string for i in pocket_res if len(i.get_heavy_atoms) == len(RESIDUES_TOPO[i.name])
        )
        return pocket_block, ligand.mol_block()

    @staticmethod
    def _split_pocket_with_surface_atoms(
        protein: Chain | Protein, ligand: Ligand, dist_cutoff: float
    ) -> tuple[str, str]:
        protein.compute_surface_atoms()

        res = np.array(protein.get_residues)
        cm_res = np.array([r.center_of_mass for r in res])

        lig_conformer = ligand.mol.GetConformer()
        lig_pos = np.array(
            [
                [
                    lig_conformer.GetAtomPosition(a.GetIdx()).x,
                    lig_conformer.GetAtomPosition(a.GetIdx()).y,
                    lig_conformer.GetAtomPosition(a.GetIdx()).z,
                ]
                for a in ligand.mol.GetAtoms()
            ]
        )
        dist_mat = compute_dist_mat(lig_pos, cm_res)
        bool_dist_mat = dist_mat < dist_cutoff
        pocket_res = res[bool_dist_mat.sum(axis=0) > 0]
        pocket_block: list[str] = []
        for r in pocket_res:
            for a in r.get_heavy_atoms:
                if a.is_surf is True:
                    pocket_block.append(a.to_string + " surf")
                else:
                    pocket_block.append(a.to_string + " inner")
        pocket_block_str = "\n".join(pocket_block)
        return pocket_block_str, ligand.mol_block()

    def _do_split(self, items: list[str]) -> None:
        try:
            sub_path = items[4].split("/")[0]
            ligand_name = items[4].split("/")[1].split(".")[0]
            chain_id = items[3].split(".")[0].split("_")[-2]

            protein = Protein(str(self.main_path / self.sample_path / items[3]))
            chain = protein.get_chain(chain_id)
            ligand = Ligand(str(self.main_path / self.sample_path / items[4]), sanitize=True)

            if self.get_surface_atom:
                pocket_block, ligand_block = self._split_pocket_with_surface_atoms(
                    chain, ligand, self.dist_cutoff
                )
            else:
                pocket_block, ligand_block = self._split_pocket(chain, ligand, self.dist_cutoff)
            save_path = self.main_path / self.new_sample_path / sub_path
            save_path.mkdir(parents=True, exist_ok=True)
            pocket_file_name = save_path / f"{ligand_name}_pocket{self.dist_cutoff}.pdb"

            with open(pocket_file_name, "w") as f:
                f.write(pocket_block)
            with open(save_path / f"{ligand_name}.mol", "w") as f:
                f.write(ligand_block)
        except Exception:
            protein_file = self.main_path / self.sample_path / items[3]
            ligand_file = self.main_path / self.sample_path / items[4]
            print("[Exception]", protein_file, ligand_file)

    @staticmethod
    def split_pocket_from_site_map(site_map: Path | str, protein_file: Path | str, dist_cutoff: float) -> str:
        site_coords: list[list[float]] = []
        site_map_path = Path(site_map)
        protein_file_path = Path(protein_file)
        with open(site_map_path) as fr:
            lines = fr.readlines()
            for line in lines:
                if line.startswith("HETATM"):
                    line = line.strip()
                    xyz = [float(line[30:38].strip()), float(line[38:46].strip()), float(line[46:54].strip())]
                    site_coords.append(xyz)
        site_coords_arr = np.array(site_coords)

        protein = Protein(str(protein_file_path))
        res = np.array(protein.get_residues)
        cm_res = np.array([r.center_of_mass for r in res])
        dist_mat = compute_dist_mat(site_coords_arr, cm_res)
        bool_dist_mat = dist_mat < dist_cutoff
        pocket_res = res[bool_dist_mat.sum(axis=0) > 0]
        pocket_block = "\n".join(
            i.to_heavy_string for i in pocket_res if len(i.get_heavy_atoms) == len(RESIDUES_TOPO[i.name])
        )
        return pocket_block

    def __call__(self, num_processes: int = 10) -> list[None]:
        pool = Pool(processes=num_processes)
        data_pool = pool.map(self._do_split, self.types)
        pool.close()
        pool.join()
        print("Done !")
        return data_pool
