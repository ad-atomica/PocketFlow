from __future__ import annotations

import unittest
from unittest.mock import patch

from rdkit import Chem
from rdkit.Chem.rdchem import Mol

from pocket_flow.utils import metrics


def _mol(smiles: str) -> Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        msg = f"Failed to parse SMILES: {smiles}"
        raise ValueError(msg)
    return mol


class TestPatternHelpers(unittest.TestCase):
    def test_has_fused4ring_skips_none_and_matches(self) -> None:
        mol_match = _mol("C1CC1")
        mol_no = _mol("CC")
        pattern = Chem.MolFromSmarts("C1CC1")
        self.assertIsNotNone(pattern)
        with patch.object(metrics, "FUSED_QUA_RING_PATTERN", [None, pattern]):
            self.assertTrue(metrics.has_fused4ring(mol_match))
            self.assertFalse(metrics.has_fused4ring(mol_no))

    def test_judge_unexpected_ring_matches(self) -> None:
        mol_match = _mol("C1CCC1")
        mol_no = _mol("C1CCCCC1")
        pattern = Chem.MolFromSmarts("C1CCC1")
        self.assertIsNotNone(pattern)
        with patch.object(metrics, "PATTERNS_1", [None, pattern]):
            self.assertTrue(metrics.judge_unexpected_ring(mol_match))
            self.assertFalse(metrics.judge_unexpected_ring(mol_no))

    def test_judge_fused_ring_uses_patterns(self) -> None:
        mol_match = _mol("C1CCCCC1")
        mol_no = _mol("C1CCC1")
        pattern = Chem.MolFromSmarts("C1CCCCC1")
        self.assertIsNotNone(pattern)
        with patch.object(metrics, "PATTERNS", [pattern]):
            with patch.object(metrics, "FUSED_QUA_RING_PATTERN", [None]):
                self.assertTrue(metrics.judge_fused_ring(mol_match))
                self.assertFalse(metrics.judge_fused_ring(mol_no))


class TestSubstructure(unittest.TestCase):
    def test_substructure_counts_and_rates(self) -> None:
        mol3 = _mol("C1CC1")
        mol4 = _mol("C1CCC1")
        mol5 = _mol("C1CCCC1")
        mol6 = _mol("C1CCCCC1")
        mol7 = _mol("C1CCCCCC1")
        mol8 = _mol("C1CCCCCCC1")
        mol9 = _mol("C1CCCCCCCC1")
        decalin = _mol("C1CCC2CCCCC2C1")
        acyclic = _mol("CC")

        fused_pat = Chem.MolFromSmarts("C1CCCCCC1")
        unexpected_pat = Chem.MolFromSmarts("C1CCC1")
        self.assertIsNotNone(fused_pat)
        self.assertIsNotNone(unexpected_pat)

        def _is_fused(mol: Mol) -> bool:
            return mol.HasSubstructMatch(fused_pat)

        def _is_unexpected(mol: Mol) -> bool:
            return mol.HasSubstructMatch(unexpected_pat)

        mol_lib = [
            [mol3, mol4, None],
            [mol5, mol6, mol7],
            [mol8, mol9, decalin],
            [acyclic],
        ]

        with patch.object(metrics, "judge_fused_ring", new=_is_fused):
            with patch.object(metrics, "judge_unexpected_ring", new=_is_unexpected):
                stats = metrics.substructure(mol_lib)

        self.assertEqual(stats["tri_ring"]["num"], 1)
        self.assertEqual(stats["qua_ring"]["num"], 1)
        self.assertEqual(stats["fif_ring"]["num"], 1)
        self.assertEqual(stats["hex_ring"]["num"], 2)
        self.assertEqual(stats["hep_ring"]["num"], 1)
        self.assertEqual(stats["oct_ring"]["num"], 1)
        self.assertEqual(stats["big_ring"]["num"], 1)
        self.assertEqual(stats["fused_ring"]["num"], 1)
        self.assertEqual(stats["unexpected_ring"]["num"], 1)

        self.assertAlmostEqual(stats["tri_ring"].get("rate", 0.0), 0.1)
        self.assertAlmostEqual(stats["qua_ring"].get("rate", 0.0), 0.1)
        self.assertAlmostEqual(stats["fif_ring"].get("rate", 0.0), 0.1)
        self.assertAlmostEqual(stats["hex_ring"].get("rate", 0.0), 0.2)
        self.assertAlmostEqual(stats["hep_ring"].get("rate", 0.0), 0.1)
        self.assertAlmostEqual(stats["oct_ring"].get("rate", 0.0), 0.1)
        self.assertAlmostEqual(stats["big_ring"].get("rate", 0.0), 0.1)
        self.assertAlmostEqual(stats["fused_ring"].get("rate", 0.0), 0.1)
        self.assertAlmostEqual(stats["unexpected_ring"].get("rate", 0.0), 0.1)

        sssr = stats["sssr"]
        expected_sssr: dict[int, int] = {}
        total_num = sum(len(batch) for batch in mol_lib)
        for batch in mol_lib:
            for mol in batch:
                if mol is None:
                    continue
                normalized = Chem.MolFromSmiles(Chem.MolToSmiles(mol, isomericSmiles=False))
                if normalized is None:
                    continue
                count = len(Chem.GetSSSR(normalized))
                expected_sssr[count] = expected_sssr.get(count, 0) + 1

        self.assertEqual(
            {count: entry["num"] for count, entry in sssr.items()},
            expected_sssr,
        )
        for count, num in expected_sssr.items():
            self.assertAlmostEqual(sssr[count].get("rate", 0.0), num / total_num)

    def test_substructure_empty_raises(self) -> None:
        with self.assertRaises(ZeroDivisionError):
            metrics.substructure([[]])


class TestSmoothing(unittest.TestCase):
    def test_smoothing_weight_zero_returns_input(self) -> None:
        scalars = [1.0, 2.0, 3.0]
        self.assertEqual(metrics.smoothing(scalars, weight=0.0), scalars)

    def test_smoothing_weight_one_is_constant(self) -> None:
        scalars = [1.0, 2.0, 3.0]
        self.assertEqual(metrics.smoothing(scalars, weight=1.0), [1.0, 1.0, 1.0])

    def test_smoothing_values(self) -> None:
        scalars = [1.0, 2.0, 3.0]
        expected = [1.0, 1.5, 2.25]
        self.assertEqual(metrics.smoothing(scalars, weight=0.5), expected)

    def test_smoothing_empty_raises(self) -> None:
        with self.assertRaises(IndexError):
            metrics.smoothing([])
