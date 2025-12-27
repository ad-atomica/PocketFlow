"""
Ring- and substructure-based metrics used for dataset inspection/evaluation.

The functions in this module compute coarse statistics about ring sizes and a
few hand-crafted "special" ring motifs (e.g., certain fused ring patterns) for a
library of RDKit molecules.

Key conventions:

- Inputs are RDKit `Mol` objects (or `None` placeholders). This module treats
  the input as a *library* and reports per-molecule presence counts: each ring
  size bucket is incremented at most once per molecule, even if a molecule has
  multiple rings of that size.
- Before ring analysis, each molecule is round-tripped through a non-isomeric
  SMILES representation (`isomericSmiles=False`) to drop stereochemistry and
  normalize the RDKit internal representation.
- Several SMARTS patterns are precompiled at import time. If RDKit fails to
  parse a SMARTS pattern, the corresponding entry will be `None` and is skipped
  during matching.
"""

from collections.abc import Sequence
from typing import NotRequired, TypedDict

from rdkit import Chem
from rdkit.Chem.rdchem import Mol


class RingCountEntry(TypedDict):
    """Count (and optional rate) for a category in :func:`substructure`.

    Keys:
        num: Number of molecules matching the category.
        rate: Optional fraction of molecules matching the category. When present,
            it is computed as `num / total_num`, where `total_num` is the number
            of items across all sublists passed to :func:`substructure`.
    """

    num: int
    rate: NotRequired[float]


class RingSizeStats(TypedDict):
    """Return type for :func:`substructure`.

    Each top-level key maps to a :class:`RingCountEntry` describing how many
    molecules in the input library match that category.

    Keys:
        tri_ring: Molecules containing at least one 3-membered ring.
        qua_ring: Molecules containing at least one 4-membered ring.
        fif_ring: Molecules containing at least one 5-membered ring.
        hex_ring: Molecules containing at least one 6-membered ring.
        hep_ring: Molecules containing at least one 7-membered ring.
        oct_ring: Molecules containing at least one 8-membered ring.
        big_ring: Molecules containing any ring larger than 8 members.
        fused_ring: Molecules matching any of the fused-ring SMARTS patterns in
            :data:`PATTERNS` or :data:`FUSED_QUA_RING_PATTERN`.
        unexpected_ring: Molecules matching any of the "unexpected ring" SMARTS
            patterns in :data:`PATTERNS_1`.
        sssr: Mapping from RDKit's `SSSR` ring count (as returned by
            `Chem.GetSSSR`) to a :class:`RingCountEntry` describing how many
            molecules had that SSSR value.
    """

    tri_ring: RingCountEntry
    qua_ring: RingCountEntry
    fif_ring: RingCountEntry
    hex_ring: RingCountEntry
    hep_ring: RingCountEntry
    oct_ring: RingCountEntry
    big_ring: RingCountEntry
    fused_ring: RingCountEntry
    unexpected_ring: RingCountEntry
    sssr: dict[int, RingCountEntry]


FUSED_QUA_RING_PATTERN: list[Mol | None] = [
    Chem.MolFromSmarts(i)
    for i in [
        "[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@[R]1~&@[R](~&@[R]~&@[R]3~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@3~&@[R]~&@1)~&@[R]~&@2",
        "[R]12~&@[R]~&@[R]~&@[R]~&@[R]3~&@[R]~&@1~&@[R](~&@[R]~&@[R]~&@[R]~&@2)~&@[R]~&@[R]1~&@[R]~&@3~&@[R]~&@[R]~&@[R]~&@[R]~&@1",
        "[R]12~&@[R]3~&@[R]~&@[R]~&@[R]4~&@[R]~&@[R]~&@[R]~&@[R](~&@[R]~&@1~&@4)~&@[R]~&@[R]~&@[R]~&@2~&@[R]~&@[R]~&@[R]~&@3",
        "[R]12~&@[R]3~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@3~&@[R]3~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@3~&@[R]~&@1~&@[R]~&@[R]~&@[R]~&@[R]~&@2",
        "[R]12~&@[R]~&@[R]~&@[R]3~&@[R](~&@[R]~&@[R]~&@[R]4~&@[R]~&@3~&@[R]~&@[R]~&@[R]~&@[R]~&@4)~&@[R]~&@1~&@[R]~&@[R]~&@[R]~&@[R]~&@2",
        "[R]12~&@[R]~&@[R]~&@[R]3~&@[R]~&@[R]4~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@4~&@[R]~&@[R]~&@3~&@[R]~&@1~&@[R]~&@[R]~&@[R]~&@[R]~&@2",
        "[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@[R]1~&@[R]~&@[R]3~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@3~&@[R]~&@[R]~&@1~&@[R]~&@2",
        "[R]12~&@[R]~&@[R]3~&@[R](~&@[R]~&@[R]4~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@4~&@[R]~&@3)~&@[R]~&@1~&@[R]~&@[R]~&@[R]~&@[R]~&@2",
        "[R]12~&@[R]~&@[R]~&@[R]3~&@[R]~&@[R]4~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@4~&@[R]~&@[R]~&@3~&@[R]~&@1~&@[R]~&@[R]~&@[R]~&@2",
        "[R]12~&@[R]~&@[R]3~&@[R]~&@[R]4~&@[R](~&@[R]~&@[R]~&@[R]~&@[R]~&@4)~&@[R]~&@3~&@[R]~&@1~&@[R]~&@[R]~&@[R]~&@[R]~&@2",
        "[R]12~&@[R]~&@[R]3~&@[R]4~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@4~&@[R]~&@[R]~&@3~&@[R]~&@1~&@[R]~&@[R]~&@[R]~&@[R]~&@2",
        "[R]12~&@[R]~&@[R]~&@[R]3~&@[R]4~&@[R]~&@[R]~&@[R]~&@[R]~&@4~&@[R]~&@[R]~&@[R]~&@3~&@[R]~&@1~&@[R]~&@[R]~&@[R]~&@2",
        "[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]1~&@[R](~&@[R]~&@[R]~&@[R]~&@1)~&@[R]1~&@[R]~&@2~&@[R]~&@[R]~&@[R]~&@1",
        "[R]12~&@[R]~&@[R]~&@[R]3~&@[R]~&@[R]~&@[R]4~&@[R](~&@[R]~&@[R]~&@[R]~&@4)~&@[R]~&@3~&@[R]~&@1~&@[R]~&@[R]~&@[R]~&@2",
        "[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@[R]~&@[R]1~&@[R]3~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@3~&@[R]~&@[R]~&@1~&@[R]~&@2",
        "[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]1~&@[R]~&@[R]3~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@3~&@[R]~&@[R]~&@1~&@[R]~&@[R]~&@[R]~&@2",
        "[R]12~&@[R]~&@[R]~&@[R]~&@[R]3~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@3~&@[R]~&@1~&@[R]~&@[R]~&@[R]~&@2",
        "[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@[R]~&@[R]1~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@2",
        "[R]12~&@[R]~&@[R]3~&@[R](~&@[R]~&@[R]~&@[R]4~&@[R]~&@3~&@[R]~&@[R]~&@[R]~&@4)~&@[R]~&@1~&@[R]~&@[R]~&@[R]~&@[R]~&@2",
        "[R]12~&@[R]~&@[R]3~&@[R](~&@[R]~&@[R]4~&@[R](~&@[R]~&@[R]~&@[R]~&@4)~&@[R]~&@3)~&@[R]~&@1~&@[R]~&@[R]~&@[R]~&@[R]~&@2",
        "[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]3~&@[R]~&@1~&@[R]1~&@[R](~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@[R]~&@3)~&@[R]~&@[R]~&@2",
        "[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]3~&@[R]~&@1~&@[R]1~&@[R](~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@[R]~&@3)~&@[R]~&@[R]~&@2",
        "[R]12~&@[R]~&@[R]~&@[R]~&@[R]3~&@[R]~&@1~&@[R]1~&@[R](~&@[R]~&@[R]~&@2)~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@[R]~&@3",
        "[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]3~&@[R]~&@1~&@[R](~&@[R]~&@[R]~&@3)~&@[R]~&@[R]~&@[R]~&@2",
        "[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@[R]1~&@[R]3~&@[R]~&@2~&@[R]~&@[R]~&@[R]~&@3~&@[R]~&@[R]~&@[R]~&@1",
        "[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]3~&@[R]~&@1~&@[R](~&@[R]1~&@[R](~&@[R]~&@[R]~&@[R]~&@1)~&@[R]~&@2)~&@[R]~&@[R]~&@[R]~&@3",
        "[R]12~&@[R]~&@[R]~&@[R]3~&@[R]~&@1~&@[R]1~&@[R](~&@[R]~&@[R]~&@2)~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@[R]~&@3",
        "[R]12~&@[R]~&@[R]3~&@[R]~&@[R]~&@[R]4~&@[R]~&@3~&@[R](~&@[R]~&@[R]~&@[R]~&@4)~&@[R]~&@1~&@[R]~&@[R]~&@[R]~&@2",
        "[R]12~&@[R]~&@[R]~&@[R]3~&@[R]~&@1~&@[R](~&@[R]~&@[R]~&@[R]~&@2)~&@[R]~&@[R]1~&@[R]~&@3~&@[R]~&@[R]~&@[R]~&@1",
        "[R]12~&@[R]~&@[R]~&@[R]~&@[R]3~&@[R]~&@1~&@[R](~&@[R]~&@[R]~&@[R]~&@[R]~&@2)~&@[R]~&@[R]~&@[R]~&@3",
        "[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@[R]1~&@[R]~&@2~&@[R]~&@[R]~&@[R]2~&@[R]~&@1~&@[R]~&@[R]~&@[R]~&@[R]~&@2",
        "[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@[R]1~&@[R]~&@2~&@[R]2~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@2~&@[R]~&@[R]~&@1",
        "[R]12~&@[R]~&@[R]~&@[R]3~&@[R]~&@1~&@[R](~&@[R]~&@[R]~&@[R]~&@[R]~&@2)~&@[R]~&@[R]~&@3",
        "[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@[R]1~&@[R]3~&@[R]~&@2~&@[R]~&@[R]~&@[R]~&@[R]~&@3~&@[R]~&@[R]~&@1",
        "[R]12~&@[R]~&@[R]~&@[R]~&@[R]34~&@[R]~&@1~&@[R](~&@[R]~&@[R]~&@[R]~&@2)~&@[R]~&@[R]~&@3~&@[R]~&@[R]~&@[R]~&@[R]~&@4",
        "[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@[R]1~&@[R](~&@[R]~&@[R]3~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@3~&@[R]~&@1)~&@[R]~&@2",
        "[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@[R]1~&@[R](~&@[R]3~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@3~&@[R]~&@[R]~&@1)~&@[R]~&@2",
        "[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]1~&@[R](~&@[R]~&@[R]~&@[R]~&@[R]~&@1)~&@[R]1~&@[R]~&@2~&@[R]~&@[R]~&@[R]~&@[R]~&@1",
        "[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]1~&@[R]3~&@[R](~&@[R]~&@[R]~&@[R]~&@[R]~&@3~&@[R]~&@[R]~&@[R]~&@1)~&@[R]~&@2",
        "[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@[R]1~&@[R](~&@[R]~&@[R]3~&@[R]~&@[R]~&@[R]~&@[R]~&@3~&@[R]~&@1)~&@[R]~&@2",
        "[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@[R]1~&@[R](~&@[R]~&@[R]~&@[R]3~&@[R]~&@1~&@[R]~&@[R]~&@[R]~&@3)~&@[R]~&@2",
        "[R]12~&@[R]~&@[R]~&@[R]3~&@[R]4~&@[R]~&@1~&@[R](~&@[R]~&@[R]~&@4~&@[R]~&@[R]~&@[R]~&@3)~&@[R]~&@[R]~&@[R]~&@2",
        "[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]1~&@[R]3~&@[R](~&@[R]~&@[R]~&@[R]~&@[R]~&@3~&@[R]~&@[R]~&@1)~&@[R]~&@2",
        "[R]12~&@[R]~&@[R]~&@[R]~&@[R]3~&@[R](~&@[R]~&@[R]4~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@4~&@[R]~&@3)~&@[R]~&@1~&@[R]~&@[R]~&@[R]~&@[R]~&@2",
        "[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@[R]1~&@[R]~&@[R]3~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@3~&@[R]~&@[R]~&@1~&@[R]~&@[R]~&@2",
        "[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@[R]1~&@[R]3~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@3~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@[R]~&@2",
        "[R]12~&@[R]~&@[R]~&@[R]~&@[R]3~&@[R](~&@[R]~&@[R]~&@[R]~&@[R]~&@3)~&@[R]~&@1~&@[R]~&@[R]~&@[R]1~&@[R]~&@2~&@[R]~&@[R]~&@[R]~&@[R]~&@1",
        "[R]12~&@[R]~&@[R]~&@[R]~&@[R]3~&@[R@@H](~&@[R@H]4~&@[R]~&@[R]~&@[R]~&@[R]~&@[R@H]~&@4~&@[R]~&@[R]~&@3)~&@[R]~&@1~&@[R]~&@[R]~&@[R]~&@[R]~&@2",
        "[R]12~&@[R]~&@[R]3~&@[R](~&@[R]~&@[R]~&@[R]~&@[R]~&@3)~&@[R]3~&@[R](~&@[R]~&@[R]~&@[R]~&@[R]~&@3)~&@[R]~&@1~&@[R]~&@[R]~&@[R]~&@[R]~&@2",
        "[R]12~&@[R]~&@[R]3~&@[R](~&@[R]~&@[R]~&@[R]~&@[R]~&@3)~&@[R]3~&@[R](~&@[R]~&@[R]~&@[R]~&@3)~&@[R]~&@1~&@[R]~&@[R]~&@[R]~&@[R]~&@2",
        "[R]12~&@[R]3~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@3~&@[R]~&@[R]3~&@[R]~&@[R]~&@[R]~&@[R]~&@3~&@[R]~&@1~&@[R]~&@[R]~&@[R]~&@[R]~&@2",
        "[R]12~&@[R]~&@[R]~&@[R]~&@[R]3~&@[R]~&@1~&@[R](~&@[R]~&@[R]~&@[R]~&@2)~&@[R]1~&@[R]~&@3~&@[R]~&@[R]~&@[R]~&@[R]~&@1",
        "[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@[R]1~&@[R](~&@[R]3~&@[R]~&@[R]~&@[R]~&@[R]~&@3~&@[R]~&@1)~&@[R]~&@2",
        "[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@[R]1~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@[R]~&@2",
        "[R]12~&@[R]~&@[R]~&@[R]~&@[R]3~&@[R](~&@[R]~&@[R]~&@[R]~&@[R]~&@3)~&@[R]~&@1~&@[R]~&@[R]~&@[R]~&@[R]~&@2",
        "[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]1~&@[R]~&@[R]~&@[R]~&@[R]3~&@[R]~&@1~&@[R](~&@[R]~&@[R]~&@[R]~&@3)~&@[R]~&@2",
        "[R]12~&@[R]3~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@3~&@[R]3~&@[R]~&@[R]~&@[R]~&@[R]~&@3~&@[R]~&@1~&@[R]~&@[R]~&@[R]~&@[R]~&@2",
        "[R]12~&@[R]~&@[R]~&@[R]3~&@[R]~&@1~&@[R](~&@[R]1~&@[R](~&@[R]~&@2)~&@[R]~&@[R]~&@[R]~&@[R]~&@1)~&@[R]~&@[R]~&@[R]~&@3",
        "[R]12~&@[R]~&@[R]~&@[R]3~&@[R](~&@[R]~&@[R]~&@[R]~&@[R]~&@3)~&@[R]~&@1~&@[R]~&@[R]~&@[R]1~&@[R]~&@2~&@[R]~&@[R]~&@[R]~&@1",
        "[R]12~&@[R]~&@[R]~&@[R]3~&@[R](~&@[R]~&@[R]~&@[R]~&@[R]~&@3)~&@[R]~&@1~&@[R]~&@[R]1~&@[R](~&@[R]~&@[R]~&@[R]~&@1)~&@[R]~&@2",
        "[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]1~&@[R](~&@[R]~&@[R]~&@[R]3~&@[R]~&@1~&@[R]~&@[R]~&@[R]~&@3)~&@[R]~&@[R]~&@2",
        "[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@[R]1~&@[R]~&@[R]3~&@[R]~&@[R]~&@[R]~&@[R]~&@3~&@[R]~&@[R]~&@1~&@[R]~&@2",
        "[R]12~&@[R]~&@[R]~&@[R]3~&@[R](~&@[R]~&@[R]4~&@[R]~&@[R]~&@[R]~&@[R]~&@4~&@[R]~&@3)~&@[R]~&@1~&@[R]~&@[R]~&@[R]~&@2",
        "[R]12~&@[R]~&@[R]~&@[R]3~&@[R]~&@[R]~&@[R]4~&@[R](~&@[R]~&@[R]~&@[R]~&@[R]~&@4)~&@[R]~&@3~&@[R]~&@1~&@[R]~&@[R]~&@[R]~&@[R]~&@2",
        "[R]12~&@[R]~&@[R]~&@[R]3~&@[R]4~&@[R](~&@[R]~&@[R]~&@[R]~&@3)~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@4~&@[R]~&@[R]~&@[R]~&@[R]~&@2",
        "[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@13~&@[R](~&@[R]~&@[R]~&@[R]~&@3)~&@[R]~&@[R]1~&@[R]~&@2~&@[R]~&@[R]~&@[R]~&@[R]~&@1",
        "[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@[R]1~&@[R]~&@2~&@[R]2~&@[R]~&@[R]~&@[R]~&@[R]~&@2~&@[R]~&@[R]~&@1",
    ]
]


def has_fused4ring(mol: Mol) -> bool:
    """Check whether `mol` contains any of the pre-defined fused 4-ring motifs.

    Args:
        mol: RDKit molecule to test.

    Returns:
        `True` if any SMARTS pattern in :data:`FUSED_QUA_RING_PATTERN` matches
        `mol`, otherwise `False`.
    """
    for pat in FUSED_QUA_RING_PATTERN:
        if pat is not None and mol.HasSubstructMatch(pat):
            return True
    return False


PATTERNS_1: list[Mol | None] = [
    Chem.MolFromSmarts(i)
    for i in [
        "[R]12~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@2",
        "[R]12~&@[R]~&@[R]~&@1~&@[R]~&@[R]~&@[R]~&@2",
        "[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@2",
        "[R]12~&@[R]~&@[R]~&@1~&@[R]~&@2",
        "[R]12~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@[R]~&@[R]~&@[R]~&@2",
        "[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@[R]~&@2",
        "[R]12~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@[R]~&@2",
        "[R]1~&@[R]~&@[R]~&@12~&@[R]~&@[R]~&@2",
    ]
]


def judge_unexpected_ring(mol: Mol) -> bool:
    """Heuristic detector for "unexpected" ring motifs.

    This checks whether `mol` contains any SMARTS in :data:`PATTERNS_1`.

    Notes:
        - This helper uses `mol.GetSubstructMatches` (not just a boolean match)
          and returns `True` if at least one match is found.

    Args:
        mol: RDKit molecule to test.

    Returns:
        `True` if any pattern in :data:`PATTERNS_1` matches `mol`, otherwise
        `False`.
    """
    for pat in PATTERNS_1:
        if pat is not None:
            subs = mol.GetSubstructMatches(pat)
            if len(subs) > 0:
                return True
    return False


PATTERNS: list[Mol | None] = [
    Chem.MolFromSmarts(i)
    for i in [
        "[R]12~&@[R]3~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@3~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@[R]~&@[R]~&@[R]~&@2",
        "[R]12~&@[R]~&@[R]3~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@3~&@[R]~&@[R]~&@1~&@[R]~&@[R]~&@[R]~&@2",
        "[R]12~&@[R]~&@[R]~&@[R]~&@[R]3~&@[R]~&@1~&@[R](~&@[R]~&@[R]~&@[R]~&@2)~&@[R]~&@[R]~&@[R]~&@3",
        "[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]1~&@[R](~&@[R]~&@2)~&@[R]~&@[R]~&@[R]~&@[R]~&@1",
        "[R]12~&@[R]~&@[R]~&@[R]3~&@[R](~&@[R]~&@1~&@[R]~&@[R]~&@[R]~&@[R]~&@2)~&@[R]~&@[R]~&@[R]~&@3",
        "[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@[R]1~&@[R](~&@[R]~&@2)~&@[R]~&@[R]~&@[R]~&@[R]~&@1",
    ]
]


def judge_fused_ring(mol: Mol) -> bool:
    """Heuristic detector for fused-ring motifs.

    Args:
        mol: RDKit molecule to test.

    Returns:
        `True` if any SMARTS pattern in :data:`PATTERNS` or
        :data:`FUSED_QUA_RING_PATTERN` matches `mol`, otherwise `False`.
    """
    for pat in PATTERNS + FUSED_QUA_RING_PATTERN:
        if pat is not None and mol.HasSubstructMatch(pat):
            return True
    return False


def substructure(mol_lib: Sequence[Sequence[Mol | None]]) -> RingSizeStats:
    """Compute ring-size and ring-motif statistics for a molecule library.

    The input is treated as a nested collection (e.g., a list of batches). Each
    molecule contributes at most 1 count to each ring-size bucket, regardless of
    how many rings of that size it contains.

    The function also returns a histogram over RDKit's `SSSR` ring count
    (`Chem.GetSSSR`).

    Notes:
        - `total_num` is computed as the sum of the lengths of the nested
          sequences in `mol_lib` (including `None` placeholders). Numerators
          only count valid molecules (non-`None` and successfully normalized),
          so rates may be < 1.0 even if many entries are `None`.
        - Each molecule is normalized via
          `Chem.MolFromSmiles(Chem.MolToSmiles(mol, isomericSmiles=False))`
          before ring analysis; if normalization fails, that molecule is skipped.

    Args:
        mol_lib: Nested sequences of RDKit molecules (or `None` placeholders).

    Returns:
        A :class:`RingSizeStats` dictionary with `num` and `rate` values for each
        category.

    Raises:
        ZeroDivisionError: If `mol_lib` is empty (or contains only empty
            sublists), since rates are computed as `num / total_num`.
    """
    total_num = sum(len(s) for s in mol_lib)
    tri_ring: RingCountEntry = {"num": 0}
    qua_ring: RingCountEntry = {"num": 0}
    fif_ring: RingCountEntry = {"num": 0}
    hex_ring: RingCountEntry = {"num": 0}
    hep_ring: RingCountEntry = {"num": 0}
    oct_ring: RingCountEntry = {"num": 0}
    big_ring: RingCountEntry = {"num": 0}
    fused_ring: RingCountEntry = {"num": 0}
    unexpected_ring: RingCountEntry = {"num": 0}
    sssr_dict: dict[int, RingCountEntry] = {}

    for s in mol_lib:
        for mol in s:
            if mol is None:
                continue
            mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol, isomericSmiles=False))
            if mol is None:
                continue
            sssr = Chem.GetSSSR(mol)
            if sssr in sssr_dict:
                sssr_dict[sssr]["num"] += 1
            else:
                sssr_dict[sssr] = {"num": 1}

            has_ring_size_3 = False
            has_ring_size_4 = False
            has_ring_size_5 = False
            has_ring_size_6 = False
            has_ring_size_7 = False
            has_ring_size_8 = False
            has_big_ring = False
            has_fused_ring = False
            has_unexpected_ring = False
            for r in mol.GetRingInfo().AtomRings():
                ring_len = len(r)
                if ring_len == 3 and not has_ring_size_3:
                    has_ring_size_3 = True
                    tri_ring["num"] += 1
                if ring_len == 4 and not has_ring_size_4:
                    has_ring_size_4 = True
                    qua_ring["num"] += 1
                if ring_len == 5 and not has_ring_size_5:
                    has_ring_size_5 = True
                    fif_ring["num"] += 1
                if ring_len == 6 and not has_ring_size_6:
                    has_ring_size_6 = True
                    hex_ring["num"] += 1
                if ring_len == 7 and not has_ring_size_7:
                    has_ring_size_7 = True
                    hep_ring["num"] += 1
                if ring_len == 8 and not has_ring_size_8:
                    has_ring_size_8 = True
                    oct_ring["num"] += 1
                if ring_len > 8 and not has_big_ring:
                    has_big_ring = True
                    big_ring["num"] += 1
            if judge_fused_ring(mol) and not has_fused_ring:
                has_fused_ring = True
                fused_ring["num"] += 1
            if judge_unexpected_ring(mol) and not has_unexpected_ring:
                unexpected_ring["num"] += 1

    tri_ring["rate"] = tri_ring["num"] / total_num
    qua_ring["rate"] = qua_ring["num"] / total_num
    fif_ring["rate"] = fif_ring["num"] / total_num
    hex_ring["rate"] = hex_ring["num"] / total_num
    hep_ring["rate"] = hep_ring["num"] / total_num
    oct_ring["rate"] = oct_ring["num"] / total_num
    big_ring["rate"] = big_ring["num"] / total_num
    fused_ring["rate"] = fused_ring["num"] / total_num
    unexpected_ring["rate"] = unexpected_ring["num"] / total_num
    for k in sssr_dict:
        sssr_dict[k]["rate"] = sssr_dict[k]["num"] / total_num

    return {
        "tri_ring": tri_ring,
        "qua_ring": qua_ring,
        "fif_ring": fif_ring,
        "hex_ring": hex_ring,
        "hep_ring": hep_ring,
        "oct_ring": oct_ring,
        "big_ring": big_ring,
        "fused_ring": fused_ring,
        "unexpected_ring": unexpected_ring,
        "sssr": sssr_dict,
    }


def smoothing(scalars: Sequence[float], weight: float = 0.8) -> list[float]:
    """Compute an exponential moving average (EMA) over a scalar sequence.

    This is a lightweight smoothing utility commonly used for plotting noisy
    curves (e.g., training loss).

    The recurrence is:
        `s_t = weight * s_{t-1} + (1 - weight) * x_t`, with `s_0 = x_0`.

    Args:
        scalars: Input values. Must be non-empty.
        weight: EMA weight in the range `[0, 1]`. Larger values produce heavier
            smoothing (slower response).

    Returns:
        A list of smoothed values with the same length as `scalars`.

    Raises:
        IndexError: If `scalars` is empty (the implementation accesses
            `scalars[0]`).
    """
    last = scalars[0]
    smoothed: list[float] = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed
