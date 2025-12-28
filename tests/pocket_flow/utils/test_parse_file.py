from __future__ import annotations

import importlib.util
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
from rdkit import Chem
from rdkit.Chem import AllChem

from pocket_flow.utils.ParseFile import (
    AMINO_ACID_TYPE,
    ATOM_FAMILIES,
    BACKBONE_SYMBOL,
    BOND_TYPE_ID,
    Atom,
    Chain,
    Ligand,
    Protein,
    Residue,
    parse_sdf_to_dict,
)
from pocket_flow.utils.residues_base import RESIDUES_TOPO


def _pdb_atom_line(
    idx: int,
    name: str,
    res_name: str,
    chain: str,
    res_idx: int,
    x: float,
    y: float,
    z: float,
    occupancy: float = 1.0,
    temp: float = 10.0,
    seg_id: str = "",
    element: str | None = None,
    tail: str | None = None,
) -> str:
    """Generate a PDB-format ATOM line with specified coordinates and metadata."""
    if element is None:
        element = name.strip()[0]
    fmt = (
        "{:6s}{:5d} {:^4s}{:1s}{:3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}        {:<2s}{:2s}"
    )
    line = fmt.format(
        "ATOM",
        idx,
        name,
        "",
        res_name,
        chain,
        res_idx,
        "",
        x,
        y,
        z,
        occupancy,
        temp,
        seg_id,
        element,
    )
    if tail:
        line = f"{line} {tail}"
    return line


def _make_residue_lines(
    res_name: str,
    chain: str,
    res_idx: int,
    atom_names: list[str],
    start_idx: int = 1,
    annotate_surf: bool = False,
    surf_names: set[str] | None = None,
) -> tuple[list[str], int]:
    """Generate PDB ATOM lines for a residue with optional surface annotation."""
    lines: list[str] = []
    idx = start_idx
    for offset, atom_name in enumerate(atom_names):
        tail = None
        if annotate_surf:
            tail = "surf" if surf_names and atom_name in surf_names else "inner"
        line = _pdb_atom_line(
            idx=idx,
            name=atom_name,
            res_name=res_name,
            chain=chain,
            res_idx=res_idx,
            x=float(offset),
            y=float(offset + 0.1),
            z=float(offset + 0.2),
            tail=tail,
        )
        lines.append(line)
        idx += 1
    return lines, idx


class TestAtom(unittest.TestCase):
    """Tests for Atom class parsing and attribute extraction."""

    def test_atom_missing_element_field_raises(self) -> None:
        line = _pdb_atom_line(
            idx=1,
            name="CA",
            res_name="ALA",
            chain="A",
            res_idx=1,
            x=1.0,
            y=2.0,
            z=3.0,
            element="",
        )
        with self.assertRaises(ValueError) as ctx:
            Atom(line)
        self.assertIn("Missing element symbol", str(ctx.exception))

    def test_atom_trailing_spaces_preserve_element(self) -> None:
        line = _pdb_atom_line(
            idx=1,
            name="CA",
            res_name="ALA",
            chain="A",
            res_idx=1,
            x=1.0,
            y=2.0,
            z=3.0,
            element="C",
        )
        atom = Atom(f"{line}   ")
        self.assertEqual(atom.element, "C")
        self.assertFalse(atom.is_surf)

    def test_atom_missing_element_columns_raises(self) -> None:
        line = _pdb_atom_line(
            idx=1,
            name="CA",
            res_name="ALA",
            chain="A",
            res_idx=1,
            x=1.0,
            y=2.0,
            z=3.0,
            element="C",
        )
        line = line[:76]
        with self.assertRaises(ValueError) as ctx:
            Atom(line)
        self.assertIn("Missing element symbol", str(ctx.exception))
        self.assertIn("columns 77-78", str(ctx.exception))

    def test_atom_invalid_element_field_raises(self) -> None:
        line = _pdb_atom_line(
            idx=1,
            name="CA",
            res_name="ALA",
            chain="A",
            res_idx=1,
            x=1.0,
            y=2.0,
            z=3.0,
            element="Xx",
        )
        with self.assertRaises(ValueError) as ctx:
            Atom(line)
        self.assertIn("Invalid element symbol", str(ctx.exception))

    def test_atom_parsing_selenium_surface_flags(self) -> None:
        """Tests that selenium (MSE) atoms are correctly converted to methionine (MET)
        with proper element mapping, disorder flags, and surface annotation."""
        line = _pdb_atom_line(
            idx=1,
            name="SE",
            res_name="MSE",
            chain="A",
            res_idx=7,
            x=1.0,
            y=2.0,
            z=3.0,
            occupancy=0.5,
            element="SE",
            tail="surf",
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            atom = Atom(line)
        self.assertEqual(len(caught), 1)
        self.assertIn("MSE", str(caught[0].message))
        self.assertIn("MET", str(caught[0].message))
        self.assertEqual(atom.element, "S")
        self.assertEqual(atom.name, "SD")
        self.assertEqual(atom.res_name, "MET")
        self.assertTrue(atom.is_disorder)
        self.assertTrue(atom.is_surf)
        self.assertEqual(atom.to_dict["element"], 16)
        self.assertEqual(atom.to_dict["atom_to_aa_type"], AMINO_ACID_TYPE["MET"])
        self.assertFalse(atom.to_dict["is_backbone"])
        self.assertEqual(atom.to_string[76:78].strip(), "S")

    def test_atom_parsing_selenocysteine_mapping_warning(self) -> None:
        """Tests that selenium (SEC) atoms are correctly converted to cysteine (CYS)
        with proper element mapping and warning emission."""
        line = _pdb_atom_line(
            idx=2,
            name="SE",
            res_name="SEC",
            chain="B",
            res_idx=42,
            x=4.0,
            y=5.0,
            z=6.0,
            element="SE",
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            atom = Atom(line)
        self.assertEqual(len(caught), 1)
        self.assertIn("SEC", str(caught[0].message))
        self.assertIn("CYS", str(caught[0].message))
        self.assertEqual(atom.element, "S")
        self.assertEqual(atom.name, "SG")
        self.assertEqual(atom.res_name, "CYS")
        self.assertEqual(atom.to_dict["element"], 16)
        self.assertEqual(atom.to_dict["atom_to_aa_type"], AMINO_ACID_TYPE["CYS"])


class TestResidue(unittest.TestCase):
    """Tests for Residue class bond graph construction and atom handling."""

    def test_residue_bond_graph_and_duplicate_atoms(self) -> None:
        """Tests that residue bond graphs match expected topology and duplicate atoms
        are handled correctly by creating a residue with duplicate CB atoms and verifying
         bond connectivity."""
        atom_names = list(RESIDUES_TOPO["ALA"].keys())
        lines, idx = _make_residue_lines("ALA", "A", 1, atom_names, start_idx=1)
        duplicate_cb = _pdb_atom_line(
            idx=idx,
            name="CB",
            res_name="ALA",
            chain="A",
            res_idx=1,
            x=10.0,
            y=11.0,
            z=12.0,
        )
        lines.append(duplicate_cb)
        residue = Residue(lines)

        self.assertFalse(residue.is_disorder)
        self.assertTrue(residue.is_perfect)
        self.assertEqual(len(residue.get_heavy_atoms), len(RESIDUES_TOPO["ALA"]))

        edge_index, edge_type = residue.bond_graph
        self.assertEqual(edge_index.shape[0], 2)
        self.assertEqual(edge_type.shape[0], edge_index.shape[1])

        atom_order = [a.name for a in residue.get_heavy_atoms]
        name_to_idx = {name: ix for ix, name in enumerate(atom_order)}
        expected_pairs: dict[tuple[int, int], int] = {}
        for src, neighbors in RESIDUES_TOPO["ALA"].items():
            for dst, bond_t in neighbors.items():
                expected_pairs[(name_to_idx[src], name_to_idx[dst])] = bond_t


class TestLigand(unittest.TestCase):
    """Tests for Ligand class SDF/MOL parsing and coordinate normalization."""

    def test_empty_dict_contract(self) -> None:
        empty = Ligand.empty_dict()
        self.assertEqual(empty.element.shape, (0,))
        self.assertEqual(empty.element.dtype, np.int64)
        self.assertEqual(empty.pos.shape, (0, 3))
        self.assertEqual(empty.pos.dtype, np.float32)
        self.assertEqual(empty.bond_index.shape, (2, 0))
        self.assertEqual(empty.bond_index.dtype, np.int64)
        self.assertEqual(empty.bond_type.shape, (0,))
        self.assertEqual(empty.bond_type.dtype, np.int64)
        self.assertEqual(empty.center_of_mass.shape, (3,))
        self.assertEqual(empty.center_of_mass.dtype, np.float32)
        assert_allclose(empty.center_of_mass, np.zeros((3,), dtype=np.float32))
        self.assertEqual(empty.atom_feature.shape, (0, len(ATOM_FAMILIES)))
        self.assertEqual(empty.atom_feature.dtype, np.int64)
        self.assertEqual(empty.ring_info, {})
        self.assertIsNone(empty.filename)

    def test_parse_sdf_and_normalize_pos(self) -> None:
        """Tests that SDF files are correctly parsed into dictionaries with proper
        bond indices and atom features, and ligand coordinate normalization applies
         the correct transformation by creating an ethanol molecule and verifying coordinate
        shifts."""
        if not importlib.util.find_spec("rdkit"):
            self.skipTest("requires rdkit")

        mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        AllChem.EmbedMolecule(mol, randomSeed=0xF00D)
        AllChem.UFFOptimizeMolecule(mol)
        mol_no_h = Chem.RemoveHs(mol)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            sdf_path = tmp_path / "ethanol.sdf"
            mol_path = tmp_path / "ethanol.mol"

            writer = Chem.SDWriter(str(sdf_path))
            writer.write(mol_no_h)
            writer.close()
            Chem.MolToMolFile(mol_no_h, str(mol_path))

            lig_dict = parse_sdf_to_dict(str(sdf_path))
            num_atoms = mol_no_h.GetNumAtoms()
            num_bonds = mol_no_h.GetNumBonds()
            self.assertEqual(lig_dict["element"].shape[0], num_atoms)
            self.assertEqual(lig_dict["bond_index"].shape, (2, num_bonds * 2))
            self.assertEqual(lig_dict["bond_type"].shape[0], num_bonds * 2)
            self.assertTrue(np.all(lig_dict["bond_type"] == BOND_TYPE_ID["SINGLE"]))
            self.assertEqual(lig_dict["atom_feature"].shape, (num_atoms, len(ATOM_FAMILIES)))
            self.assertEqual(set(lig_dict["ring_info"].keys()), set(range(num_atoms)))
            for ring_vec in lig_dict["ring_info"].values():
                self.assertEqual(ring_vec.size, 0)

            ligand = Ligand(str(mol_path))
            self.assertEqual(ligand.name, "ethanol")
            ligand_dict = ligand.to_dict()
            self.assertEqual(ligand_dict["element"].shape[0], num_atoms)
            self.assertEqual(ligand_dict["bond_index"].shape, (2, num_bonds * 2))
            self.assertTrue(np.all(ligand_dict["bond_type"] == BOND_TYPE_ID["SINGLE"]))

            conformer = ligand.mol.GetConformer()
            coords_before = np.array(
                [
                    (
                        conformer.GetAtomPosition(a.GetIdx()).x,
                        conformer.GetAtomPosition(a.GetIdx()).y,
                        conformer.GetAtomPosition(a.GetIdx()).z,
                    )
                    for a in ligand.mol.GetAtoms()
                ]
            )
            shift = np.array([1.5, -2.0, 0.5], dtype=np.float64)
            ligand.normalize_pos(shift, np.eye(3))
            coords_after = np.array(
                [
                    (
                        conformer.GetAtomPosition(a.GetIdx()).x,
                        conformer.GetAtomPosition(a.GetIdx()).y,
                        conformer.GetAtomPosition(a.GetIdx()).z,
                    )
                    for a in ligand.mol.GetAtoms()
                ]
            )
            assert_allclose(coords_after, coords_before - shift, rtol=1e-6, atol=1e-6)
            assert ligand.normalized_coords is not None
            assert_allclose(ligand.normalized_coords, coords_after, rtol=1e-6, atol=1e-6)

    def test_single_atom_ligand_has_empty_bond_graph(self) -> None:
        """A single-atom molecule should export an empty (2,0) bond_index and (0,) bond_type."""
        if not importlib.util.find_spec("rdkit"):
            self.skipTest("requires rdkit")

        from rdkit.Geometry import Point3D

        mol = Chem.MolFromSmiles("C")
        conf = Chem.Conformer(mol.GetNumAtoms())
        conf.SetAtomPosition(0, Point3D(0.0, 0.0, 0.0))
        mol.AddConformer(conf, assignId=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            sdf_path = tmp_path / "single_atom.sdf"
            mol_path = tmp_path / "single_atom.mol"

            writer = Chem.SDWriter(str(sdf_path))
            writer.write(mol)
            writer.close()
            Chem.MolToMolFile(mol, str(mol_path))

            lig_dict = parse_sdf_to_dict(str(sdf_path))
            self.assertEqual(lig_dict["element"].shape, (1,))
            self.assertEqual(lig_dict["bond_index"].shape, (2, 0))
            self.assertEqual(lig_dict["bond_index"].dtype, np.int64)
            self.assertEqual(lig_dict["bond_type"].shape, (0,))
            self.assertEqual(lig_dict["bond_type"].dtype, np.int64)

            ligand = Ligand(str(mol_path))
            ligand_dict = ligand.to_dict()
            self.assertEqual(ligand_dict["element"].shape, (1,))
            self.assertEqual(ligand_dict["bond_index"].shape, (2, 0))
            self.assertEqual(ligand_dict["bond_index"].dtype, np.int64)
            self.assertEqual(ligand_dict["bond_type"].shape, (0,))
            self.assertEqual(ligand_dict["bond_type"].dtype, np.int64)


class TestChain(unittest.TestCase):
    """Tests for Chain class residue filtering and incomplete residue handling."""

    def test_chain_filters_incomplete_residues(self) -> None:
        """Tests that chains correctly filter incomplete residues based on ignore_incomplete_res
        flag by creating a chain with one complete and one incomplete residue."""
        atom_names = list(RESIDUES_TOPO["ALA"].keys())
        lines_complete, idx = _make_residue_lines("ALA", "A", 1, atom_names, start_idx=1)
        lines_incomplete, _ = _make_residue_lines("ALA", "A", 2, atom_names[:-1], start_idx=idx)
        chain_info = {1: lines_complete, 2: lines_incomplete}

        chain = Chain(chain_info, ignore_incomplete_res=True)
        self.assertEqual(len(chain.get_residues), 1)
        self.assertEqual(len(chain.get_incomplete_residues), 1)
        self.assertEqual(len(chain.get_heavy_atoms), len(RESIDUES_TOPO["ALA"]))

        chain_all = Chain(chain_info, ignore_incomplete_res=False)
        self.assertEqual(len(chain_all.get_residues), 2)


class TestProtein(unittest.TestCase):
    """Tests for Protein class dictionary generation, peptide bond detection, and surface atom handling."""

    def test_empty_pdb_exports_empty_bond_graph(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            pdb_path = tmp_path / "empty.pdb"
            pdb_path.write_text("")

            protein = Protein(str(pdb_path))
            atom_dict = protein.get_atom_dict()
            self.assertEqual(atom_dict["element"].shape, (0,))
            self.assertEqual(atom_dict["pos"].shape, (0, 3))
            self.assertEqual(atom_dict["bond_index"].shape, (2, 0))
            self.assertEqual(atom_dict["bond_index"].dtype, np.int64)
            self.assertEqual(atom_dict["bond_type"].shape, (0,))
            self.assertEqual(atom_dict["bond_type"].dtype, np.int64)

    def test_protein_dicts_and_peptide_bond(self) -> None:
        """Tests that protein atom dictionaries include surface masks, peptide bonds are correctly
        identified between residues, and backbone dictionaries contain only backbone atoms
        by parsing a two-residue PDB file."""
        atom_names = list(RESIDUES_TOPO["ALA"].keys())
        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_path = Path(tmpdir) / "tiny.pdb"
            lines: list[str] = []
            lines1, idx = _make_residue_lines(
                "ALA",
                "A",
                1,
                atom_names,
                start_idx=1,
                annotate_surf=True,
                surf_names={"CB"},
            )
            lines.extend(lines1)
            lines2, _ = _make_residue_lines(
                "ALA",
                "A",
                2,
                atom_names,
                start_idx=idx,
                annotate_surf=True,
                surf_names={"N"},
            )
            lines.extend(lines2)
            pdb_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

            protein = Protein(str(pdb_path), ignore_incomplete_res=True)
            self.assertTrue(protein.has_surf_atom)

            atom_dict = protein.get_atom_dict(get_surf=True)
            self.assertEqual(atom_dict["pos"].shape, (10, 3))
            self.assertEqual(atom_dict["element"].dtype, np.int64)
            assert "surface_mask" in atom_dict
            surface_mask = atom_dict["surface_mask"]
            self.assertTrue(surface_mask.any())
            self.assertTrue(np.logical_not(surface_mask).any())

            atom_names_list = atom_dict["atom_name"]
            c_idx_res1 = atom_names_list.index("C")
            n_idx_res2 = atom_names_list.index("N", 5)
            edges = {tuple(atom_dict["bond_index"][:, k]) for k in range(atom_dict["bond_index"].shape[1])}
            self.assertIn((c_idx_res1, n_idx_res2), edges)
            self.assertIn((n_idx_res2, c_idx_res1), edges)

            backbone_dict = protein.get_backbone_dict()
            self.assertEqual(len(backbone_dict["atom_name"]), 8)
            self.assertTrue(all(name in BACKBONE_SYMBOL for name in backbone_dict["atom_name"]))
            self.assertEqual(backbone_dict["bond_index"].shape, (2, 0))

    def test_peptide_bond_with_out_of_order_atoms(self) -> None:
        atom_names = list(RESIDUES_TOPO["ALA"].keys())
        atom_names_res1 = [name for name in atom_names if name not in {"N", "C"}] + ["N", "C"]
        atom_names_res2 = ["C"] + [name for name in atom_names if name not in {"C", "N"}] + ["N"]

        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_path = Path(tmpdir) / "out_of_order.pdb"
            lines: list[str] = []
            lines1, idx = _make_residue_lines("ALA", "A", 1, atom_names_res1, start_idx=1)
            lines2, _ = _make_residue_lines("ALA", "A", 2, atom_names_res2, start_idx=idx)
            lines.extend(lines1)
            lines.extend(lines2)
            pdb_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

            protein = Protein(str(pdb_path), ignore_incomplete_res=True)
            atom_dict = protein.get_atom_dict()
            residues = protein.get_residues
            res1_names = [a.name for a in residues[0].get_heavy_atoms]
            res2_names = [a.name for a in residues[1].get_heavy_atoms]
            c_idx_res1 = res1_names.index("C")
            n_idx_res2 = len(res1_names) + res2_names.index("N")
            edges = {tuple(atom_dict["bond_index"][:, k]) for k in range(atom_dict["bond_index"].shape[1])}
            self.assertIn((c_idx_res1, n_idx_res2), edges)
            self.assertIn((n_idx_res2, c_idx_res1), edges)
