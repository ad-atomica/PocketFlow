"""PDB/SDF parsing utilities and lightweight structural containers.

This module provides minimal data structures and parsers used throughout
PocketFlow to load:

- **Proteins** from PDB files (``ATOM`` records), represented as :class:`Protein`,
  :class:`Chain`, :class:`Residue`, and :class:`Atom`.
- **Ligands** from MOL/SDF files via RDKit, represented as :class:`Ligand`.

The central "export" APIs are :meth:`Protein.get_atom_dict` /
:meth:`Protein.get_backbone_dict` and :meth:`Ligand.to_dict`, which produce
NumPy-based dictionaries that can be converted into the model's data objects.

Dependency notes:
    - RDKit is required for ligand parsing and chemical feature extraction.
    - PyMOL is *optional* and only needed for surface-atom annotation via
      :meth:`Protein.compute_surface_atoms` and :meth:`Chain.compute_surface_atoms`.
      If PyMOL is not importable, those methods will fail at runtime.

Conventions:
    - Coordinates are in Ångström and stored as NumPy arrays.
    - Bond indices follow a PyG-like convention: a ``(2, E)`` array of integer
      indices, with both directions explicitly included for each bond.
"""

from __future__ import annotations

import os
import warnings
from typing import TYPE_CHECKING, Any, NotRequired, TypedDict, cast

import numpy as np
from easydict import EasyDict
from numpy.typing import NDArray
from rdkit import Chem, RDConfig
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem.rdchem import BondType

from .residues_base import RESIDUES_TOPO

if TYPE_CHECKING:
    from rdkit.Chem.rdchem import Mol

try:
    import pymol
except ImportError:
    print("we can not compute the atoms on the surface of protein, because pymol can not be imported")

ATOM_TYPE_WITH_HYBIRD: list[str] = [
    "SP3_C",
    "SP2_C",
    "SP_C",
    "SP3_N",
    "SP2_N",
    "SP_N",
    "SP3_O",
    "SP2_O",
    "SP3_F",
    "SP3_P",
    "SP2_P",
    "SP3D_P",
    "SP3_S",
    "SP2_S",
    "SP3D_S",
    "SP3D2_S",
    "SP3_Cl",
    "SP3_Br",
    "SP3_I",
]
ATOM_MAP: list[int] = [6, 6, 6, 7, 7, 7, 8, 8, 9, 15, 15, 15, 16, 16, 16, 16, 17, 35, 53]
PT: Chem.rdchem.PeriodicTable = Chem.GetPeriodicTable()
BACKBONE_SYMBOL: set[str] = {"N", "CA", "C", "O"}
AMINO_ACID_TYPE: dict[str, int] = {
    "CYS": 0,
    "GLY": 1,
    "ALA": 2,
    "THR": 3,
    "LYS": 4,
    "PRO": 5,
    "VAL": 6,
    "SER": 7,
    "ASN": 8,
    "LEU": 9,
    "GLN": 10,
    "MET": 11,
    "ASP": 12,
    "TRP": 13,
    "HIS": 14,
    "GLU": 15,
    "ARG": 16,
    "ILE": 17,
    "PHE": 18,
    "TYR": 19,
}


class AtomDict(TypedDict):
    """Dictionary representation of a single protein atom.

    Keys:
        element: Atomic number (e.g., carbon = 6).
        pos: Cartesian coordinates with shape ``(3,)``.
        is_backbone: Whether this atom is a protein backbone atom (N, CA, C, O).
        atom_name: PDB atom name (e.g., ``"CA"``).
        atom_to_aa_type: Integer-encoded residue type (see :data:`AMINO_ACID_TYPE`).
    """

    element: int
    pos: NDArray[np.floating[Any]]
    is_backbone: bool
    atom_name: str
    atom_to_aa_type: int


class ProteinAtomDict(TypedDict):
    """Dictionary representation of a protein structure (heavy atoms only).

    This is the primary output format returned by :meth:`Protein.get_atom_dict`.

    Shapes/dtypes:
        - ``element``: ``(N,)`` ``int64`` atomic numbers.
        - ``pos``: ``(N, 3)`` ``float32`` coordinates.
        - ``is_backbone``: ``(N,)`` ``bool`` mask.
        - ``atom_name``: ``N``-length list of PDB atom names.
        - ``atom_to_aa_type``: ``(N,)`` ``int64`` residue-type ids.
        - ``bond_index``: ``(2, E)`` ``int64`` directed edges.
        - ``bond_type``: ``(E,)`` ``int64`` bond types (see :data:`BOND_NAMES`).
        - ``surface_mask`` (optional): ``(N,)`` ``bool`` surface-atom mask.

    Notes:
        All arrays are indexed over heavy atoms only. The bond graph and surface
        annotations are defined over heavy atoms only.
    """

    element: NDArray[np.int64]
    pos: NDArray[np.float32]
    is_backbone: NDArray[np.bool_]
    atom_name: list[str]
    atom_to_aa_type: NDArray[np.int64]
    molecule_name: str | None
    bond_index: NDArray[np.int64]
    bond_type: NDArray[np.int64]
    filename: str | None
    surface_mask: NotRequired[NDArray[np.bool_]]


class LigandDict(TypedDict):
    """Dictionary representation of a ligand molecule.

    This is the primary output format produced by :func:`parse_sdf_to_dict` and
    :meth:`Ligand.to_dict`.

    Shapes/dtypes:
        - ``element``: ``(N,)`` ``int64`` atomic numbers.
        - ``pos``: ``(N, 3)`` ``float32`` coordinates from the first conformer.
        - ``bond_index``: ``(2, E)`` ``int64`` directed edges.
        - ``bond_type``: ``(E,)`` ``int64`` bond types (see :data:`BOND_NAMES`).
        - ``center_of_mass``: ``(3,)`` ``float32``.
        - ``atom_feature``: ``(N, F)`` ``int64`` pharmacophore-family features.
        - ``ring_info``: Per-atom ring membership as returned by :func:`is_in_ring`.
    """

    element: NDArray[np.int64]
    pos: NDArray[np.float32]
    bond_index: NDArray[np.int64]
    bond_type: NDArray[np.int64]
    center_of_mass: NDArray[np.float32]
    atom_feature: NDArray[np.int64]
    ring_info: dict[int, NDArray[np.int64]]
    filename: str | None


def _map_selenium_to_sulfur(
    element: str, res_name: str, atom_name: str, res_idx: int, chain: str
) -> tuple[str, str, str]:
    """Map selenium atoms in non-standard residues to sulfur equivalents.

    Converts selenium atoms (``SE``) in selenomethionine (``MSE``) or
    selenocysteine (``SEC``) to sulfur and remaps to standard residues
    ``MET/SD`` or ``CYS/SG``. Emits a warning when mapping is applied.

    Args:
        element: Atomic element symbol (e.g., "SE").
        res_name: Residue name (e.g., "MSE", "SEC").
        atom_name: Atom name (e.g., "SE").
        res_idx: Residue index.
        chain: Chain identifier.

    Returns:
        A tuple ``(element, atom_name, res_name)`` with potentially updated
        values. If no mapping is needed, returns the original values unchanged.
    """
    if element == "SE":
        se_residue_map = {"MSE": ("MET", "SD"), "SEC": ("CYS", "SG")}
        mapped = se_residue_map.get(res_name)
        if mapped is not None:
            mapped_res_name, mapped_atom_name = mapped
            warnings.warn(
                f"Mapping selenium residue {res_name}{res_idx} chain {chain} to "
                f"{mapped_res_name} ({atom_name}->{mapped_atom_name}, SE->S).",
                UserWarning,
                stacklevel=3,
            )
            return "S", mapped_atom_name, mapped_res_name
    return element, atom_name, res_name


def _normalize_residue_name(res_name: str) -> str:
    """Normalize non-standard residue names to standard equivalents.

    Converts selenomethionine (``MSE``) to methionine (``MET``) and
    selenocysteine (``SEC``) to cysteine (``CYS``).

    Args:
        res_name: Residue name (e.g., "MSE", "SEC").

    Returns:
        Normalized residue name. Returns the original name if no mapping exists.
    """
    residue_map = {"MSE": "MET", "SEC": "CYS"}
    return residue_map.get(res_name, res_name)


class Atom:
    """Single PDB ``ATOM`` record parsed into a structured object.

    Instances are constructed from a single fixed-width PDB line. Only the
    canonical PDB columns are used; any trailing tokens (e.g., ``"surf"`` /
    ``"inner"``) are parsed separately to set :attr:`is_surf`.

    Special-casing:
        - Selenium atoms (``SE``) in selenomethionine (``MSE``) or
          selenocysteine (``SEC``) are converted to sulfur and remapped to the
          standard residues ``MET/SD`` or ``CYS/SG``. A warning is emitted when
          this mapping is applied to non-standard residues.
    """

    idx: int
    name: str
    res_name: str
    chain: str
    res_idx: int
    coord: NDArray[np.floating[Any]]
    occupancy: float
    temperature_factor: float
    seg_id: str
    element: str
    mass: float
    is_disorder: bool
    is_surf: bool

    def __init__(self, atom_info: str) -> None:
        """Parse a PDB ``ATOM`` record into fields.

        Args:
            atom_info: A single PDB line starting with ``"ATOM"``. This is
                expected to be in fixed-width PDB format (see
                https://cupnet.net/pdb-format/). If the last whitespace-delimited
                token equals ``"surf"``, :attr:`is_surf` is set to ``True``.

        Raises:
            ValueError: If numeric fields (atom serial, residue id, coordinates,
                occupancy, etc.) cannot be parsed.
        """
        self.idx = int(atom_info[6:11])
        self.name = atom_info[12:16].strip()
        self.res_name = atom_info[17:20].strip()
        self.chain = atom_info[21:22].strip()
        self.res_idx = int(atom_info[22:26])
        self.coord = np.array(
            [
                float(atom_info[30:38].strip()),
                float(atom_info[38:46].strip()),
                float(atom_info[46:54].strip()),
            ]
        )
        self.occupancy = float(atom_info[54:60])
        self.temperature_factor = float(atom_info[60:66].strip())
        self.seg_id = atom_info[72:76].strip()
        self.element = atom_info[76:78].strip()
        if not self.element:
            msg = (
                "Missing element symbol in PDB ATOM record columns 77-78 "
                f"(idx={self.idx}, atom_name={self.name!r}, res_name={self.res_name!r}, "
                f"chain={self.chain!r}, res_idx={self.res_idx})."
            )
            raise ValueError(msg)
        self.element, self.name, self.res_name = _map_selenium_to_sulfur(
            self.element, self.res_name, self.name, self.res_idx, self.chain
        )
        try:
            atomic_number = PT.GetAtomicNumber(self.element)
        except Exception as exc:  # pragma: no cover - rdkit error surface varies by version
            msg = (
                f"Invalid element symbol {self.element!r} in PDB ATOM record columns 77-78 "
                f"(idx={self.idx}, atom_name={self.name!r}, res_name={self.res_name!r}, "
                f"chain={self.chain!r}, res_idx={self.res_idx})."
            )
            raise ValueError(msg) from exc
        if atomic_number <= 0:
            msg = (
                f"Invalid element symbol {self.element!r} in PDB ATOM record columns 77-78 "
                f"(idx={self.idx}, atom_name={self.name!r}, res_name={self.res_name!r}, "
                f"chain={self.chain!r}, res_idx={self.res_idx})."
            )
            raise ValueError(msg)
        self.mass = PT.GetAtomicWeight(self.element)
        if self.occupancy < 1.0:
            self.is_disorder = True
        else:
            self.is_disorder = False
        self.is_surf = atom_info.split()[-1] == "surf"

    @property
    def to_string(self) -> str:
        """Render this atom back into a PDB-format ``ATOM`` line.

        Returns:
            A fixed-width PDB ``ATOM`` record. Surface annotations (``surf`` /
            ``inner``) are not appended; callers that require those tags should
            add them separately.
        """
        # https://cupnet.net/pdb-format/
        fmt = (  # record, serial, name, alt, res, chain, seq, insert
            "{:6s}{:5d} {:^4s}{:1s}{:3s} {:1s}{:4d}{:1s}"
            # x, y, z, occupancy, temp_factor
            + "   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}"
            # segment_id, element, charge
            + "        {:<2s}{:2s}"
        )
        out = fmt.format(
            "ATOM",
            self.idx,
            self.name,
            "",
            self.res_name,
            self.chain,
            self.res_idx,
            "",
            self.coord[0],
            self.coord[1],
            self.coord[2],
            self.occupancy,
            self.temperature_factor,
            self.seg_id,
            self.element,
        )
        return out

    @property
    def to_dict(self) -> AtomDict:
        """Convert to a compact dictionary form used by downstream pipelines."""
        return {
            "element": PT.GetAtomicNumber(self.element),
            "pos": self.coord,
            "is_backbone": self.name in BACKBONE_SYMBOL,
            "atom_name": self.name,
            "atom_to_aa_type": AMINO_ACID_TYPE[self.res_name],
        }

    def __repr__(self) -> str:
        info = (
            f"name={self.name}, index={self.idx}, res={self.res_name + str(self.res_idx)},"
            + f" chain={self.chain}, is_disorder={self.is_disorder}"
        )
        return f"{self.__class__.__name__}({info})"


class Residue:
    """A residue aggregated from multiple PDB ``ATOM`` records.

    A :class:`Residue` is initialized from the list of PDB lines belonging to a
    residue (same residue index and chain). Duplicate atom names are skipped.

    The residue is marked "perfect" if its number of heavy atoms matches the
    expected heavy-atom topology in :data:`pocket_flow.utils.residues_base.RESIDUES_TOPO`.
    """

    res_info: list[str]
    atom_dict: dict[str, Atom]
    is_disorder: bool
    idx: int
    chain: str
    name: str
    is_perfect: bool

    def __init__(self, res_info: list[str]) -> None:
        """Create a residue from its PDB ``ATOM`` lines.

        Args:
            res_info: List of PDB ``ATOM`` lines for the residue.

        Notes:
            - Disordered atoms are detected via occupancy < 1.0 on any atom.
            - Unknown residue names (not present in :data:`RESIDUES_TOPO`) will
              raise a ``KeyError`` when evaluating :attr:`is_perfect`.
        """
        self.res_info = res_info
        self.atom_dict = {}
        disorder: list[bool] = []
        for i in res_info:
            atom = Atom(i)
            if atom.name in self.atom_dict:
                continue
            else:
                self.atom_dict[atom.name] = atom
                disorder.append(atom.is_disorder)
            atom.res_name = _normalize_residue_name(atom.res_name)

        if True in disorder:
            self.is_disorder = True
        else:
            self.is_disorder = False

        self.idx = self.atom_dict[atom.name].res_idx
        self.chain = self.atom_dict[atom.name].chain
        self.name = self.atom_dict[atom.name].res_name
        self.is_perfect = len(self.get_heavy_atoms) == len(RESIDUES_TOPO[self.name])

    @property
    def to_heavy_string(self) -> str:
        """PDB string of heavy atoms only (one line per atom)."""
        return "\n".join([a.to_string for a in self.get_heavy_atoms])

    @property
    def to_string(self) -> str:
        """PDB string of all atoms (one line per atom)."""
        return "\n".join([a.to_string for a in self.get_atoms])

    @property
    def get_coords(self) -> NDArray[np.floating[Any]]:
        """Coordinates for all atoms as an ``(N, 3)`` array."""
        return np.array([a.coord for a in self.get_atoms])

    @property
    def get_atoms(self) -> list[Atom]:
        """Atoms in this residue (including hydrogens, if present)."""
        return list(self.atom_dict.values())

    @property
    def get_heavy_atoms(self) -> list[Atom]:
        """Heavy atoms in this residue (excluding hydrogens)."""
        return [a for a in self.atom_dict.values() if a.element != "H"]

    @property
    def get_heavy_coords(self) -> NDArray[np.floating[Any]]:
        """Coordinates for heavy atoms as an ``(N, 3)`` array."""
        return np.array([a.coord for a in self.get_heavy_atoms])

    @property
    def center_of_mass(self) -> NDArray[np.floating[Any]]:
        """Center of mass computed from atomic weights (includes H if present)."""
        atom_mass = np.array([i.mass for i in self.get_atoms]).reshape(-1, 1)
        return np.sum(self.get_coords * atom_mass, axis=0) / atom_mass.sum()

    @property
    def bond_graph(self) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """Residue-internal heavy-atom bond graph from template topology.

        The graph is constructed using :data:`RESIDUES_TOPO` for the residue
        name. Only bonds between heavy atoms present in this residue are added.

        Returns:
            A tuple ``(edge_index, edge_type)`` where:
              - ``edge_index`` has shape ``(2, E)`` and includes directed edges
                for each bond (both directions).
              - ``edge_type`` has shape ``(E,)`` and stores integer bond types.
        """
        i: list[int] = []
        j: list[int] = []
        bt: list[int] = []
        res_graph = RESIDUES_TOPO[self.name]
        atom_names = [atom.name for atom in self.get_heavy_atoms]
        name_to_idx = {n: k for k, n in enumerate(atom_names)}
        for ix, name in enumerate(atom_names):
            for adj, bond_t in res_graph[name].items():
                idx_j = name_to_idx.get(adj)
                if idx_j is not None:
                    i.append(ix)
                    j.append(idx_j)
                    bt.append(bond_t)
        edge_index = np.stack([i, j]).astype(dtype=np.int64)
        bt_arr = np.array(bt, dtype=np.int64)
        return edge_index, bt_arr

    @property
    def centroid(self) -> NDArray[np.floating[Any]]:
        """Arithmetic mean of all atom coordinates."""
        return self.get_coords.mean(axis=0)

    def __repr__(self) -> str:
        info = (
            f"name={self.name}, index={self.idx}, chain={self.chain},"
            + f" is_disorder={self.is_disorder}, is_perfect={self.is_perfect}"
        )
        return f"{self.__class__.__name__}({info})"


class Chain:
    """A protein chain composed of multiple :class:`Residue` objects."""

    pdb_file: str | None
    res_dict: dict[int, Residue]
    residues: dict[int, Residue]
    chain: str
    ignore_incomplete_res: bool
    has_surf_atom: bool
    center_of_mass_shift: NDArray[np.floating[Any]] | None
    rotate_matrix: NDArray[np.floating[Any]] | None

    def __init__(
        self,
        chain_info: dict[int, list[str]],
        ignore_incomplete_res: bool = True,
        pdb_file: str | None = None,
    ) -> None:
        """Create a chain from residue-indexed PDB lines.

        Args:
            chain_info: Mapping residue index → list of PDB ``ATOM`` lines for
                that residue.
            ignore_incomplete_res: If True, :attr:`get_residues` filters out
                residues that do not match the expected heavy-atom topology.
            pdb_file: Optional source PDB file path. Required for surface-atom
                computation which creates temporary per-chain files alongside
                the source PDB.
        """
        self.pdb_file = pdb_file
        self.res_dict = {}
        self.residues = {i: Residue(chain_info[i]) for i in chain_info}
        self.chain = list(self.residues.values())[0].chain
        self.__normalized__ = False
        self.center_of_mass_shift = None
        self.rotate_matrix = None
        self.ignore_incomplete_res = ignore_incomplete_res
        self.has_surf_atom = False

    @property
    def get_incomplete_residues(self) -> list[Residue]:
        """List residues that fail the topology completeness check."""
        return [i for i in self.residues.values() if not i.is_perfect]

    @property
    def to_heavy_string(self) -> str:
        """PDB string for heavy atoms in this chain."""
        return "\n".join([res.to_heavy_string for res in self.get_residues])

    @property
    def to_string(self) -> str:
        """PDB string for all atoms in this chain."""
        return "\n".join([res.to_string for res in self.get_residues])

    @property
    def get_atoms(self) -> list[Atom]:
        """All atoms in this chain (subject to residue filtering)."""
        atoms: list[Atom] = []
        for res in self.get_residues:
            atoms.extend(res.get_atoms)
        return atoms

    @property
    def get_residues(self) -> list[Residue]:
        """Residues in this chain, optionally excluding incomplete residues."""
        if self.ignore_incomplete_res:
            return [i for i in self.residues.values() if i.is_perfect]
        else:
            return list(self.residues.values())

    @property
    def get_heavy_atoms(self) -> list[Atom]:
        """Heavy atoms in this chain (subject to residue filtering)."""
        atoms: list[Atom] = []
        for res in self.get_residues:
            atoms.extend(res.get_heavy_atoms)
        return atoms

    @property
    def get_coords(self) -> NDArray[np.floating[Any]]:
        """Coordinates for all atoms as an ``(N, 3)`` array."""
        return np.array([i.coord for i in self.get_atoms])

    @property
    def get_heavy_coords(self) -> NDArray[np.floating[Any]]:
        """Coordinates for heavy atoms as an ``(N, 3)`` array."""
        return np.array([i.coord for i in self.get_heavy_atoms])

    @property
    def center_of_mass(self) -> NDArray[np.floating[Any]]:
        """Center of mass computed from atomic weights."""
        atom_mass = np.array([i.mass for i in self.get_atoms]).reshape(-1, 1)
        return np.sum(self.get_coords * atom_mass, axis=0) / atom_mass.sum()

    @property
    def centroid(self) -> NDArray[np.floating[Any]]:
        """Arithmetic mean of all atom coordinates."""
        return self.get_coords.mean(axis=0)

    def compute_surface_atoms(self) -> None:
        """Annotate heavy atoms with a surface/inner label using PyMOL.

        This routine writes a temporary PDB file for this chain, calls PyMOL's
        surface-atom utilities, writes a second temporary file containing only
        surface atoms, and then re-parses the surface subset to mark matching
        atoms in-place as :attr:`Atom.is_surf = True`.

        Side effects:
            - Creates and deletes temporary ``*.pdb`` files in the directory of
              :attr:`pdb_file`.
            - Loads objects into PyMOL session and cleans them up after use.
            - Requires a working PyMOL installation importable as ``pymol``.

        Raises:
            ValueError: If :attr:`pdb_file` is not set.
            NameError: If PyMOL could not be imported at module import time.
        """
        if self.pdb_file is None:
            msg = "pdb_file is required to compute surface atoms"
            raise ValueError(msg)
        path, filename = os.path.split(self.pdb_file)
        chain_file_name = filename.split(".")[0] + "_" + self.chain + ".pdb"
        chain_file = os.path.join(path or ".", chain_file_name)
        save_name = None
        sele = None
        name = None
        try:
            with open(chain_file, "w") as fw:
                fw.write(self.to_heavy_string)

            pymol.cmd.load(chain_file)
            sele = chain_file_name.split(".")[0]
            pymol.cmd.remove(f"({sele}) and hydro")
            name = pymol.util.find_surface_atoms(sele=sele, _self=pymol.cmd)
            save_name = os.path.join(path or ".", sele + "-surface.pdb")
            pymol.cmd.save(save_name, (name))
            surf_protein = Protein(save_name, ignore_incomplete_res=False)
            surf_res_dict = {r.idx: r for r in surf_protein.get_residues}
            for res in self.get_residues:
                res_idx = res.idx
                if res_idx in surf_res_dict:
                    for a in surf_res_dict[res_idx].get_heavy_atoms:
                        res.atom_dict[a.name].is_surf = True
            self.has_surf_atom = True
        finally:
            # Clean up PyMOL objects and selections
            if name is not None:
                try:
                    pymol.cmd.delete(name)
                except Exception:
                    pass
            if sele is not None:
                try:
                    pymol.cmd.delete(sele)
                except Exception:
                    pass
            # Clean up temporary files
            if save_name and os.path.exists(save_name):
                try:
                    os.remove(save_name)
                except OSError:
                    pass
            if os.path.exists(chain_file):
                try:
                    os.remove(chain_file)
                except OSError:
                    pass

    def get_surf_mask(self) -> NDArray[np.bool_]:
        """Return a boolean mask over heavy atoms indicating surface atoms."""
        if self.has_surf_atom is False:
            self.compute_surface_atoms()
        return np.array([a.is_surf for a in self.get_heavy_atoms], dtype=bool)

    def get_res_by_id(self, res_id: int) -> Residue:
        """Fetch a residue by its integer residue index."""
        return self.residues[res_id]

    def __repr__(self) -> str:
        tmp = "Chain={}, NumResidues={}, NumAtoms={}, NumHeavyAtoms={}"
        info = tmp.format(
            self.chain,
            len(self.residues),
            self.get_coords.shape[0],
            self.get_heavy_coords.shape[0],
        )
        return f"{self.__class__.__name__}({info})"


class Protein:
    """Protein structure parsed from a PDB file.

    The parser is intentionally lightweight and only considers ``ATOM`` records.
    Residues are grouped by (chain id, residue index) and stored as
    :class:`Chain` instances in :attr:`chains`.
    """

    ignore_incomplete_res: bool
    name: str
    pdb_file: str
    has_surf_atom: bool
    chains: dict[str, Chain]
    center_of_mass_shift: NDArray[np.floating[Any]] | None
    rotate_matrix: NDArray[np.floating[Any]] | None

    def __init__(self, pdb_file: str, ignore_incomplete_res: bool = True) -> None:
        """Load a PDB file and build chains/residues/atoms.

        Args:
            pdb_file: Path to a PDB file.
            ignore_incomplete_res: If True, downstream accessors that use
                :attr:`Chain.get_residues` will filter out residues that do not
                match the expected heavy-atom topology.

        Notes:
            If the first line of the file ends with the token ``surf`` or
            ``inner``, :attr:`has_surf_atom` is set to True. This is used to
            indicate that the input may already contain surface annotations.
            Empty files are handled gracefully: if the file is empty or contains
            no ``ATOM`` records, :attr:`chains` will be an empty dictionary.
        """
        self.ignore_incomplete_res = ignore_incomplete_res
        self.name = pdb_file.split("/")[-1].split(".")[0]
        self.pdb_file = pdb_file
        self.has_surf_atom = False
        with open(pdb_file) as fr:
            lines = fr.readlines()
            if lines:
                first_nonempty = 0
                while first_nonempty < len(lines) and not lines[first_nonempty].strip():
                    first_nonempty += 1
                if first_nonempty:
                    warnings.warn(
                        f"{pdb_file} starts with {first_nonempty} blank line(s); "
                        "removing them before parsing.",
                        UserWarning,
                    )
                    lines = lines[first_nonempty:]
                if lines:
                    tokens = lines[0].strip().split()
                    if tokens and tokens[-1] in {"surf", "inner"}:
                        self.has_surf_atom = True
            chain_info: dict[str, dict[int, list[str]]] = {}
            for line in lines:
                if line.startswith("ATOM"):
                    line = line.rstrip("\n")
                    chain = line[21:22].strip()
                    res_idx = int(line[22:26].strip())
                    if chain not in chain_info:
                        chain_info[chain] = {}
                        chain_info[chain][res_idx] = [line]
                    elif res_idx not in chain_info[chain]:
                        chain_info[chain][res_idx] = [line]
                    else:
                        chain_info[chain][res_idx].append(line)

        self.chains = {
            c: Chain(chain_info[c], ignore_incomplete_res=ignore_incomplete_res, pdb_file=pdb_file)
            for c in chain_info
        }
        self.__normalized__ = False
        self.center_of_mass_shift = None
        self.rotate_matrix = None

    @property
    def get_incomplete_residues(self) -> list[Residue]:
        """List all incomplete residues across chains."""
        res_list: list[Residue] = []
        for i in self.chains:
            res_list += self.chains[i].get_incomplete_residues
        return res_list

    @property
    def to_heavy_string(self) -> str:
        """PDB string for heavy atoms across all chains."""
        return "\n".join([res.to_heavy_string for res in self.get_residues])

    @property
    def to_string(self) -> str:
        """PDB string for all atoms across all chains."""
        return "\n".join([res.to_string for res in self.get_residues])

    @property
    def get_residues(self) -> list[Residue]:
        """Residues across all chains, with optional completeness filtering."""
        res_list: list[Residue] = []
        for i in self.chains:
            res_list += self.chains[i].get_residues
        return res_list

    @property
    def get_atoms(self) -> list[Atom]:
        """All atoms across all chains (subject to residue filtering)."""
        atoms: list[Atom] = []
        for res in self.get_residues:
            atoms.extend(res.get_atoms)
        return atoms

    @property
    def get_heavy_atoms(self) -> list[Atom]:
        """Heavy atoms across all chains (subject to residue filtering)."""
        atoms: list[Atom] = []
        for res in self.get_residues:
            atoms.extend(res.get_heavy_atoms)
        return atoms

    @property
    def get_coords(self) -> NDArray[np.floating[Any]]:
        """Coordinates for all atoms as an ``(N, 3)`` array."""
        return np.array([i.coord for i in self.get_atoms])

    @property
    def get_heavy_coords(self) -> NDArray[np.floating[Any]]:
        """Coordinates for heavy atoms as an ``(N, 3)`` array."""
        return np.array([i.coord for i in self.get_heavy_atoms])

    @property
    def center_of_mass(self) -> NDArray[np.floating[Any]]:
        """Center of mass computed from atomic weights."""
        atom_mass = np.array([i.mass for i in self.get_atoms]).reshape(-1, 1)
        return np.sum(self.get_coords * atom_mass, axis=0) / atom_mass.sum()

    @property
    def bond_graph(self) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """Protein heavy-atom bond graph (intra-residue + peptide bonds).

        Intra-residue bonds are taken from each residue's template topology via
        :meth:`Residue.bond_graph`. For consecutive residues within the same
        chain whose residue indices differ by 1, a peptide bond is added between
        the terminal ``C`` atom of the previous residue and the ``N`` atom of
        the current residue.

        Returns:
            A tuple ``(edge_index, edge_type)`` with directed edges, where
            ``edge_index`` has shape ``(2, E)`` and ``edge_type`` has shape
            ``(E,)``.
        """
        res_list = self.get_residues
        if not res_list:
            return np.empty((2, 0), dtype=np.int64), np.empty((0,), dtype=np.int64)
        bond_index: list[NDArray[np.int64]] = []
        bond_type: list[NDArray[np.int64]] = []
        N_term_list: list[int] = []
        C_term_list: list[int] = []
        cusum = 0
        for ix, res in enumerate(res_list):
            e_idx, e_type = res.bond_graph
            bond_index.append(e_idx + cusum)
            bond_type.append(e_type)
            # Indices must be computed *within the residue* (then offset by
            # cusum). Using a protein-global `.index("N")` is incorrect when
            # PDB atom ordering differs across residues.
            res_atom_names = [a.name for a in res.get_heavy_atoms]
            name_to_local_idx = {n: k for k, n in enumerate(res_atom_names)}
            try:
                N_term_ix = cusum + name_to_local_idx["N"]
                C_term_ix = cusum + name_to_local_idx["C"]
            except KeyError as e:
                raise ValueError(
                    f"Residue {res.name}{res.idx} chain {res.chain} missing backbone atom {e.args[0]!r}; "
                    "cannot build peptide-bond graph."
                ) from e
            N_term_list.append(N_term_ix)
            C_term_list.append(C_term_ix)
            cusum += res.get_heavy_coords.shape[0]
            if ix != 0:
                if res.idx - res_list[ix - 1].idx == 1 and res.chain == res_list[ix - 1].chain:
                    bond_idx_between_res = np.array(
                        [[N_term_ix, C_term_list[ix - 1]], [C_term_list[ix - 1], N_term_ix]],
                        dtype=np.int64,
                    )
                    bond_index.append(bond_idx_between_res)
                    bond_type_between_res = np.array(
                        [BOND_TYPE_ID["SINGLE"], BOND_TYPE_ID["SINGLE"]], dtype=np.int64
                    )
                    bond_type.append(bond_type_between_res)
        if not bond_index:
            return np.empty((2, 0), dtype=np.int64), np.empty((0,), dtype=np.int64)
        bond_index_arr = np.concatenate(bond_index, axis=1)
        bond_type_arr = np.concatenate(bond_type)
        return bond_index_arr, bond_type_arr

    @property
    def centroid(self) -> NDArray[np.floating[Any]]:
        """Arithmetic mean of all atom coordinates."""
        return self.get_coords.mean(axis=0)

    def get_chain(self, chain_id: str) -> Chain:
        """Fetch a chain by its chain identifier (single-character PDB chain id)."""
        return self.chains[chain_id]

    def get_res_by_id(self, chain_id: str, res_id: int) -> Residue:
        """Fetch a residue by chain id and residue index."""
        return self.chains[chain_id].get_res_by_id(res_id)

    def get_atom_dict(self, get_surf: bool = False) -> ProteinAtomDict:
        """Export the protein to a NumPy dictionary for model consumption.

        This method only supports heavy atoms (hydrogens are excluded). The bond
        graph and surface annotations are defined over heavy atoms only.

        Args:
            get_surf: If True, include a ``surface_mask`` key. If surface atoms
                have not yet been computed, callers should invoke
                :meth:`compute_surface_atoms` beforehand or call
                :meth:`get_surf_mask`.

        Returns:
            A :class:`ProteinAtomDict` containing atom-level features and a bond
            graph.
        """
        atom_dict: dict[str, Any] = {
            "element": [],
            "pos": [],
            "is_backbone": [],
            "atom_name": [],
            "atom_to_aa_type": [],
        }
        for a in self.get_atoms:
            if a.element == "H":
                continue
            atom_dict["element"].append(a.to_dict["element"])
            atom_dict["pos"].append(a.to_dict["pos"])
            atom_dict["is_backbone"].append(a.to_dict["is_backbone"])
            atom_dict["atom_name"].append(a.to_dict["atom_name"])
            atom_dict["atom_to_aa_type"].append(a.to_dict["atom_to_aa_type"])
        atom_dict["element"] = np.array(atom_dict["element"], dtype=np.int64)
        atom_dict["pos"] = np.array(atom_dict["pos"], dtype=np.float32).reshape(-1, 3)
        atom_dict["is_backbone"] = np.array(atom_dict["is_backbone"], dtype=bool)
        if get_surf:
            atom_dict["surface_mask"] = np.array([a.is_surf for a in self.get_heavy_atoms], dtype=bool)

        atom_dict["atom_to_aa_type"] = np.array(atom_dict["atom_to_aa_type"], dtype=np.int64)
        atom_dict["molecule_name"] = None
        protein_bond_index, protein_bond_type = self.bond_graph
        atom_dict["bond_index"] = protein_bond_index
        atom_dict["bond_type"] = protein_bond_type
        atom_dict["filename"] = self.pdb_file
        return cast(ProteinAtomDict, atom_dict)

    def get_backbone_dict(self) -> dict[str, Any]:
        """Export backbone-only atoms as a dictionary.

        This helper filters the output of :meth:`get_atom_dict` to only include
        atoms whose PDB atom name is one of ``{"N","CA","C","O"}``. Only heavy
        atoms are included (hydrogens are excluded).

        Returns:
            A dictionary containing backbone atom arrays and metadata. This
            helper does not currently include bond arrays.
        """
        atom_dict = self.get_atom_dict()
        backbone_dict: dict[str, Any] = {}
        backbone_dict["element"] = atom_dict["element"][atom_dict["is_backbone"]]
        backbone_dict["pos"] = atom_dict["pos"][atom_dict["is_backbone"]]
        backbone_dict["is_backbone"] = np.ones(atom_dict["is_backbone"].sum(), dtype=bool)
        backbone_dict["atom_name"] = np.array(atom_dict["atom_name"])[atom_dict["is_backbone"]].tolist()
        backbone_dict["atom_to_aa_type"] = atom_dict["atom_to_aa_type"][atom_dict["is_backbone"]]
        backbone_dict["molecule_name"] = atom_dict["molecule_name"]
        backbone_dict["bond_index"] = np.empty([2, 0], dtype=np.int64)
        backbone_dict["bond_type"] = np.empty(0, dtype=np.int64)
        backbone_dict["filename"] = self.pdb_file
        return backbone_dict

    @property
    def get_backbone(self) -> list[Atom]:
        """Backbone atoms across all residues."""
        atoms: list[Atom] = []
        for res in self.get_residues:
            bkb = [a for a in res.get_atoms if a.name in BACKBONE_SYMBOL]
            atoms += bkb
        return atoms

    def compute_surface_atoms(self) -> None:
        """Annotate heavy atoms with surface/inner labels using PyMOL.

        This method operates similarly to :meth:`Chain.compute_surface_atoms`,
        but runs surface detection on the full PDB file. Surface atoms are
        re-parsed from a temporary file and used to mark corresponding atoms
        in-place as :attr:`Atom.is_surf = True`.

        Side effects:
            - Creates and deletes a temporary ``*-surface.pdb`` file adjacent to
              :attr:`pdb_file`.
            - Loads objects into PyMOL session and cleans them up after use.
            - Requires a working PyMOL installation importable as ``pymol``.

        Raises:
            NameError: If PyMOL could not be imported at module import time.
        """
        save_name = None
        sele = None
        name = None
        try:
            pymol.cmd.load(self.pdb_file)
            path, filename = os.path.split(self.pdb_file)
            if path == "":
                path = "./"
            sele = filename.split(".")[0]
            pymol.cmd.remove(f"({sele}) and hydro")
            name = pymol.util.find_surface_atoms(sele=sele, _self=pymol.cmd)
            save_name = path + "/" + sele + "-surface.pdb"
            pymol.cmd.save(save_name, (name))
            surf_protein = Protein(save_name, ignore_incomplete_res=False)
            surf_res_dict = {r.idx: r for r in surf_protein.get_residues}
            for res in self.get_residues:
                res_idx = res.idx
                if res_idx in surf_res_dict:
                    for a in surf_res_dict[res_idx].get_heavy_atoms:
                        res.atom_dict[a.name].is_surf = True
            self.has_surf_atom = True
        finally:
            # Clean up PyMOL objects and selections
            if name is not None:
                try:
                    pymol.cmd.delete(name)
                except Exception:
                    pass
            if sele is not None:
                try:
                    pymol.cmd.delete(sele)
                except Exception:
                    pass
            # Clean up temporary files
            if save_name and os.path.exists(save_name):
                try:
                    os.remove(save_name)
                except OSError:
                    pass

    def get_surf_mask(self) -> NDArray[np.bool_]:
        """Return a boolean mask over heavy atoms indicating surface atoms."""
        if self.has_surf_atom is False:
            self.compute_surface_atoms()
        return np.array([a.is_surf for a in self.get_heavy_atoms], dtype=bool)

    @staticmethod
    def empty_dict() -> EasyDict:
        """Create an empty protein dictionary with the expected keys/dtypes."""
        empty_pocket_dict = EasyDict()
        empty_pocket_dict.element = np.empty(0, dtype=np.int64)
        empty_pocket_dict.pos = np.empty([0, 3], dtype=np.float32)
        empty_pocket_dict.is_backbone = np.empty(0, dtype=bool)
        empty_pocket_dict.atom_name = []
        empty_pocket_dict.atom_to_aa_type = np.empty(0, dtype=np.int64)
        empty_pocket_dict.molecule_name = None
        empty_pocket_dict.bond_index = np.empty([2, 0], dtype=np.int64)
        empty_pocket_dict.bond_type = np.empty(0, dtype=np.int64)
        empty_pocket_dict.filename = None
        return empty_pocket_dict

    def __repr__(self) -> str:
        num_res = 0
        num_atom = 0
        for chain_key in self.chains:
            res_list = list(self.chains[chain_key].residues.values())
            num_res += len(res_list)
            for res in res_list:
                num_atom += len(res.get_heavy_atoms)
        num_incomp = len(self.get_incomplete_residues)
        tmp = "Name={}, NumChains={}, NumResidues={}, NumHeavyAtoms={}, NumIncompleteRes={}"
        info = tmp.format(self.name, len(self.chains), num_res, num_atom, num_incomp)
        return f"{self.__class__.__name__}({info})"


####################################################################################################################


ATOM_FAMILIES: list[str] = [
    "Acceptor",
    "Donor",
    "Aromatic",
    "Hydrophobe",
    "LumpedHydrophobe",
    "NegIonizable",
    "PosIonizable",
    "ZnBinder",
]
ATOM_FAMILIES_ID: dict[str, int] = {s: i for i, s in enumerate(ATOM_FAMILIES)}

# Bond type ids are part of the model/data contract and must remain stable across
# RDKit versions. Do not derive ids from RDKit internal ordering.
#
# Note: Protein residue topologies in `pocket_flow.utils.residues_base.RESIDUES_TOPO`
# already encode SINGLE=1, DOUBLE=2, TRIPLE=3. We extend that mapping with
# AROMATIC=4 and reserve 0 for unknown/unsupported bond types.
BOND_TYPE_ID: dict[str, int] = {
    "UNSPECIFIED": 0,
    "SINGLE": 1,
    "DOUBLE": 2,
    "TRIPLE": 3,
    "AROMATIC": 4,
}
BOND_NAMES: dict[int, str] = {v: k for k, v in BOND_TYPE_ID.items()}
BOND_TYPES: dict[BondType, int] = {
    getattr(BondType, name): bond_id for name, bond_id in BOND_TYPE_ID.items() if hasattr(BondType, name)
}


def _bond_type_to_id(bond_type: BondType) -> int:
    """Map an RDKit bond type to a stable integer id."""
    bond_id = BOND_TYPES.get(bond_type)
    if bond_id is None:
        supported = ", ".join(sorted(BOND_TYPE_ID.keys()))
        raise ValueError(
            f"Unsupported RDKit bond type {bond_type!s}. "
            f"Add it to BOND_TYPE_ID to define a stable id (currently supported: {supported})."
        )
    return bond_id


def is_in_ring(mol: Mol) -> dict[int, NDArray[np.int64]]:
    """Compute a simple per-atom ring membership encoding.

    The returned encoding is based on RDKit's symmetric SSSR ring set. For each
    atom ``a`` and each ring ``r_idx``:

    - ``r_idx + 1`` is appended if the atom is in that ring.
    - ``-a`` is appended otherwise.

    This produces a vector of length ``num_rings`` for each atom. The encoding
    is project-specific and is primarily consumed by the data pipeline.

    Args:
        mol: RDKit molecule.

    Returns:
        Mapping atom index → ``(num_rings,)`` ``int64`` array.
    """
    num_atoms = len(mol.GetAtoms())
    rings = Chem.GetSymmSSSR(mol)
    # Pre-allocate lists for each atom
    d: dict[int, list[int]] = {a: [] for a in range(num_atoms)}

    for a in range(num_atoms):
        for r_idx, ring in enumerate(rings):
            if a in ring:
                d[a].append(r_idx + 1)
            else:
                d[a].append(-a)

    # Convert to numpy arrays once per atom
    return {a: np.array(lst, dtype=np.int64) for a, lst in d.items()}


def parse_sdf_to_dict(mol_file: str) -> LigandDict:
    """Parse a ligand SDF/MOL file into a :class:`LigandDict`.

    This is a convenience wrapper that:
      - reads the first molecule from an SDF/MOL file via RDKit,
      - kekulizes the molecule,
      - extracts RDKit ChemicalFeatures (families in :data:`ATOM_FAMILIES`),
      - builds a directed bond list with integer bond types.

    Args:
        mol_file: Path to an SDF/MOL file readable by RDKit.

    Returns:
        A :class:`LigandDict` with coordinates, bond graph, and features.

    Raises:
        StopIteration: If the SDF supplier contains no molecules.
        ValueError: If RDKit fails to kekulize the molecule.
    """
    fdefName = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
    rdmol = next(iter(Chem.SDMolSupplier(mol_file, removeHs=True)))
    Chem.Kekulize(rdmol)
    ring_info = is_in_ring(rdmol)
    conformer = rdmol.GetConformer()
    feat_mat = np.zeros([rdmol.GetNumAtoms(), len(ATOM_FAMILIES)], dtype=np.int64)
    for feat in factory.GetFeaturesForMol(rdmol):
        feat_mat[feat.GetAtomIds(), ATOM_FAMILIES_ID[feat.GetFamily()]] = 1

    element: list[int] = []
    pos: list[tuple[float, float, float]] = []
    atom_mass: list[float] = []
    for a in rdmol.GetAtoms():
        element.append(a.GetAtomicNum())
        atom_pos = conformer.GetAtomPosition(a.GetIdx())
        pos.append((atom_pos.x, atom_pos.y, atom_pos.z))
        atom_mass.append(a.GetMass())
    element_arr = np.array(element, dtype=np.int64)
    pos_arr = np.array(pos, dtype=np.float32)
    atom_mass_arr = np.array(atom_mass, np.float32)
    center_of_mass = ((pos_arr * atom_mass_arr.reshape(-1, 1)).sum(0) / atom_mass_arr.sum()).astype(
        np.float32
    )

    edge_index: list[list[int]] = []
    edge_type: list[int] = []
    for b in rdmol.GetBonds():
        row = [b.GetBeginAtomIdx(), b.GetEndAtomIdx()]
        col = [b.GetEndAtomIdx(), b.GetBeginAtomIdx()]
        edge_index.extend([row, col])
        edge_type.extend([_bond_type_to_id(b.GetBondType())] * 2)
    if not edge_index:
        edge_index_arr = np.empty((2, 0), dtype=np.int64)
        edge_type_arr = np.empty((0,), dtype=np.int64)
    else:
        edge_index_arr = np.array(edge_index, dtype=np.int64)
        edge_index_perm = edge_index_arr[:, 0].argsort()
        edge_index_arr = edge_index_arr[edge_index_perm].T
        edge_type_arr = np.array(edge_type, dtype=np.int64)[edge_index_perm]

    return {
        "element": element_arr,
        "pos": pos_arr,
        "bond_index": edge_index_arr,
        "bond_type": edge_type_arr,
        "center_of_mass": center_of_mass,
        "atom_feature": feat_mat,
        "ring_info": ring_info,
        "filename": mol_file,
    }


class Ligand:
    """Ligand molecule wrapper around an RDKit :class:`rdkit.Chem.rdchem.Mol`.

    This class provides convenience methods to normalize coordinates and to
    export a NumPy dictionary representation compatible with PocketFlow's data
    pipeline.
    """

    mol: Mol
    name: str | None
    lig_file: str | None
    num_atoms: int
    normalized_coords: NDArray[np.floating[Any]] | None

    def __init__(self, mol_file: str | Mol, removeHs: bool = True, sanitize: bool = True) -> None:
        """Load a ligand from file (or wrap an existing RDKit Mol).

        Args:
            mol_file: Path to a MOL file, or an RDKit Mol object.
            removeHs: When loading from file, whether to remove hydrogens.
            sanitize: When loading from file, whether to sanitize the molecule.
                If initial parsing fails and returns ``None``, a second attempt
                is made with ``sanitize=False`` and a limited sanitization pass.

        Raises:
            ValueError: If RDKit cannot kekulize the parsed molecule.
        """
        if isinstance(mol_file, Chem.rdchem.Mol):
            mol = mol_file
            self.name = None
            self.lig_file = None
        else:
            mol = Chem.MolFromMolFile(mol_file, removeHs=removeHs, sanitize=sanitize)
            if mol is None:
                mol = Chem.MolFromMolFile(mol_file, removeHs=removeHs, sanitize=False)
                mol.UpdatePropertyCache(strict=False)
                Chem.SanitizeMol(
                    mol,
                    Chem.SanitizeFlags.SANITIZE_FINDRADICALS
                    | Chem.SanitizeFlags.SANITIZE_KEKULIZE
                    | Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
                    | Chem.SanitizeFlags.SANITIZE_SETCONJUGATION
                    | Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION
                    | Chem.SanitizeFlags.SANITIZE_SYMMRINGS,
                    catchErrors=True,
                )
            self.name = mol_file.split("/")[-1].split(".")[0]
            self.lig_file = mol_file

        Chem.Kekulize(mol)
        self.mol = mol
        self.num_atoms = len(self.mol.GetAtoms())
        self.normalized_coords = None

    def normalize_pos(
        self,
        shift_vector: NDArray[np.floating[Any]],
        rotate_matrix: NDArray[np.floating[Any]],
    ) -> None:
        """Apply an affine normalization transform to the ligand coordinates.

        Coordinates are updated in-place on the underlying RDKit conformer via:
        ``(pos - shift_vector) @ rotate_matrix``.

        Args:
            shift_vector: Translation to subtract, shape ``(3,)``.
            rotate_matrix: Rotation (or general linear) matrix, shape ``(3, 3)``.
        """
        conformer = self.mol.GetConformer()
        coords = np.array(
            [
                (
                    conformer.GetAtomPosition(a.GetIdx()).x,
                    conformer.GetAtomPosition(a.GetIdx()).y,
                    conformer.GetAtomPosition(a.GetIdx()).z,
                )
                for a in self.mol.GetAtoms()
            ]
        )
        coords = (coords - shift_vector) @ rotate_matrix
        for ix, pos in enumerate(coords):
            conformer.SetAtomPosition(ix, pos.tolist())
        self.normalized_coords = coords

    def mol_block(self) -> str:
        """Return the MOL block representation of the current RDKit Mol."""
        return Chem.MolToMolBlock(self.mol)

    def to_dict(self) -> LigandDict:
        """Export this ligand to a :class:`LigandDict`.

        Returns:
            A NumPy dictionary including atom features derived from RDKit
            ChemicalFeatures and a directed bond graph.
        """
        fdefName = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
        factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
        ring_info = is_in_ring(self.mol)
        conformer = self.mol.GetConformer()
        feat_mat = np.zeros([self.mol.GetNumAtoms(), len(ATOM_FAMILIES)], dtype=np.int64)
        for feat in factory.GetFeaturesForMol(self.mol):
            feat_mat[feat.GetAtomIds(), ATOM_FAMILIES_ID[feat.GetFamily()]] = 1

        element: list[int] = []
        pos: list[tuple[float, float, float]] = []
        atom_mass: list[float] = []
        for a in self.mol.GetAtoms():
            element.append(a.GetAtomicNum())
            atom_pos = conformer.GetAtomPosition(a.GetIdx())
            pos.append((atom_pos.x, atom_pos.y, atom_pos.z))
            atom_mass.append(a.GetMass())
        element_arr = np.array(element, dtype=np.int64)
        pos_arr = np.array(pos, dtype=np.float32)
        atom_mass_arr = np.array(atom_mass, np.float32)
        center_of_mass = ((pos_arr * atom_mass_arr.reshape(-1, 1)).sum(0) / atom_mass_arr.sum()).astype(
            np.float32
        )

        edge_index: list[list[int]] = []
        edge_type: list[int] = []
        for b in self.mol.GetBonds():
            row = [b.GetBeginAtomIdx(), b.GetEndAtomIdx()]
            col = [b.GetEndAtomIdx(), b.GetBeginAtomIdx()]
            edge_index.extend([row, col])
            edge_type.extend([_bond_type_to_id(b.GetBondType())] * 2)
        if not edge_index:
            edge_index_arr = np.empty((2, 0), dtype=np.int64)
            edge_type_arr = np.empty((0,), dtype=np.int64)
        else:
            edge_index_arr = np.array(edge_index, dtype=np.int64)
            edge_index_perm = edge_index_arr[:, 0].argsort()
            edge_index_arr = edge_index_arr[edge_index_perm].T
            edge_type_arr = np.array(edge_type, dtype=np.int64)[edge_index_perm]

        return {
            "element": element_arr,
            "pos": pos_arr,
            "bond_index": edge_index_arr,
            "bond_type": edge_type_arr,
            "center_of_mass": center_of_mass,
            "atom_feature": feat_mat,
            "ring_info": ring_info,
            "filename": self.lig_file,
        }

    @staticmethod
    def empty_dict() -> EasyDict:
        """Create an empty ligand dictionary with the expected keys/dtypes."""
        empty_ligand_dict = EasyDict()
        empty_ligand_dict.element = np.empty(0, dtype=np.int64)
        empty_ligand_dict.pos = np.empty([0, 3], dtype=np.float32)
        empty_ligand_dict.bond_index = np.empty([2, 0], dtype=np.int64)
        empty_ligand_dict.bond_type = np.empty(0, dtype=np.int64)
        # Keep output API-consistent: real ligands expose a (3,) COM; for an empty
        # ligand we define COM as the zero vector (rather than an "undefined" NaN
        # sentinel) to keep downstream code simple and stable.
        empty_ligand_dict.center_of_mass = np.zeros((3,), dtype=np.float32)
        empty_ligand_dict.atom_feature = np.empty([0, len(ATOM_FAMILIES)], dtype=np.int64)
        empty_ligand_dict.ring_info = {}
        empty_ligand_dict.filename = None
        return empty_ligand_dict

    def __repr__(self) -> str:
        tmp = "Name={}, NumAtoms={}"
        info = tmp.format(self.name, self.num_atoms)
        return f"{self.__class__.__name__}({info})"
