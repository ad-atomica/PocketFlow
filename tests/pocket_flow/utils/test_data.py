"""Tests for data utilities including ComplexData, batching, and collation functions."""

from __future__ import annotations

import unittest
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch

from pocket_flow.utils.data import (
    ComplexData,
    ComplexDataTrajectory,
    ProteinLigandDataLoader,
    batch_from_data_list,
    make_batch_collate,
    torchify_dict,
)


class _ComplexDataset(Dataset[ComplexData]):
    """Test dataset wrapper for ComplexData items."""

    def __init__(self, items: list[ComplexData]) -> None:
        """Initialize dataset with a list of ComplexData items."""
        self._items = items

    def __len__(self) -> int:
        """Return the number of items in the dataset."""
        return len(self._items)

    def __getitem__(self, idx: int) -> ComplexData:
        """Return the ComplexData item at the given index."""
        return self._items[idx]


def _basic_complex(
    *,
    protein_count: int,
    ligand_count: int,
    context_count: int,
    debug: Any | None = None,
) -> ComplexData:
    """Create a basic ComplexData instance for testing with specified counts."""
    data = ComplexData(
        protein_pos=torch.zeros((protein_count, 3), dtype=torch.float32),
        context_idx=torch.arange(context_count, dtype=torch.long),
        protein_element=torch.arange(protein_count, dtype=torch.long),
        ligand_element=torch.arange(ligand_count, dtype=torch.long),
    )
    if debug is not None:
        data.debug = debug
    return data


def _make_traj_step(
    *,
    num_context: int,
    num_protein: int,
    base_fields: dict[str, torch.Tensor] | None = None,
    step_batch_value: int = 0,
) -> ComplexData:
    """Create a ComplexData step suitable for ComplexDataTrajectory.from_steps tests."""
    if base_fields is None:
        base_fields = {
            "protein_pos": torch.zeros((num_protein, 3), dtype=torch.float32),
            "protein_atom_feature": torch.zeros((num_protein, 2), dtype=torch.float32),
            "ligand_pos": torch.zeros((2, 3), dtype=torch.float32),
            "ligand_element": torch.tensor([6, 8], dtype=torch.long),
            "ligand_bond_index": torch.empty((2, 0), dtype=torch.long),
            "ligand_bond_type": torch.empty((0,), dtype=torch.long),
            "ligand_atom_feature_full": torch.zeros((2, 16), dtype=torch.float32),
        }

    num_cpx = num_context + num_protein
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    return ComplexData(
        **base_fields,
        cpx_pos=torch.full((num_cpx, 3), float(num_cpx), dtype=torch.float32),
        cpx_feature=torch.full((num_cpx, 2), float(num_cpx), dtype=torch.float32),
        idx_ligand_ctx_in_cpx=torch.arange(num_context, dtype=torch.long),
        idx_protein_in_cpx=torch.arange(num_protein, dtype=torch.long) + num_context,
        candidate_focal_idx_in_protein=torch.tensor([0], dtype=torch.long),
        candidate_focal_label_in_protein=torch.tensor([True]),
        apo_protein_idx=torch.tensor([0], dtype=torch.long),
        focal_idx_in_context_selected=torch.tensor([0], dtype=torch.long),
        focal_idx_in_context_candidates=torch.tensor([0], dtype=torch.long),
        cpx_edge_index=edge_index,
        cpx_edge_type=torch.tensor([1], dtype=torch.long),
        cpx_edge_feature=torch.zeros((1, 1), dtype=torch.float32),
        cpx_backbone_index=torch.tensor([0], dtype=torch.long),
        focal_label=torch.tensor([1], dtype=torch.long),
        y_pos=torch.zeros((1, 3), dtype=torch.float32),
        ligand_frontier=torch.tensor([True]),
        edge_label=torch.tensor([1], dtype=torch.long),
        atom_label=torch.tensor([1], dtype=torch.long),
        edge_query_index_0=torch.tensor([0], dtype=torch.long),
        edge_query_index_1=torch.tensor([0], dtype=torch.long),
        pos_query_knn_edge_idx_0=torch.tensor([0], dtype=torch.long),
        pos_query_knn_edge_idx_1=torch.tensor([0], dtype=torch.long),
        pos_fake=torch.zeros((1, 3), dtype=torch.float32),
        pos_fake_knn_edge_idx_0=torch.tensor([0], dtype=torch.long),
        pos_fake_knn_edge_idx_1=torch.tensor([0], dtype=torch.long),
        pos_real=torch.zeros((1, 3), dtype=torch.float32),
        pos_real_knn_edge_idx_0=torch.tensor([0], dtype=torch.long),
        pos_real_knn_edge_idx_1=torch.tensor([0], dtype=torch.long),
        index_real_cps_edge_for_atten=torch.tensor([[0], [0]], dtype=torch.long),
        tri_edge_index=edge_index,
        tri_edge_feat=torch.zeros((1, 5), dtype=torch.long),
        step_batch=torch.full((num_cpx,), step_batch_value, dtype=torch.long),
    )


class TestComplexDataFromProteinLigandDicts(unittest.TestCase):
    """Tests for ComplexData creation from protein and ligand dictionaries."""

    def test_from_protein_ligand_dicts_builds_neighbors(self) -> None:
        """Tests that from_protein_ligand_dicts correctly builds neighbor lists from bond indices."""
        protein_dict = {
            "pos": torch.zeros((2, 3), dtype=torch.float32),
            "element": torch.tensor([1, 6], dtype=torch.long),
        }
        ligand_dict = {
            "element": torch.tensor([6, 8, 1], dtype=torch.long),
            "bond_index": torch.tensor(
                [
                    [0, 0, 1, 2],
                    [1, 2, 0, 1],
                ],
                dtype=torch.long,
            ),
        }

        data = ComplexData.from_protein_ligand_dicts(protein_dict, ligand_dict)

        self.assertTrue(torch.equal(data.protein_pos, protein_dict["pos"]))
        self.assertTrue(torch.equal(data.protein_element, protein_dict["element"]))
        self.assertTrue(torch.equal(data.ligand_element, ligand_dict["element"]))
        self.assertTrue(torch.equal(data.ligand_bond_index, ligand_dict["bond_index"]))
        self.assertEqual(data.ligand_nbh_list, {0: [1, 2], 1: [0], 2: [1]})

    def test_from_protein_ligand_dicts_requires_bond_index(self) -> None:
        """Tests that from_protein_ligand_dicts raises AttributeError when bond_index is missing."""
        ligand_dict = {"element": torch.tensor([6], dtype=torch.long)}
        with self.assertRaises(AttributeError):
            ComplexData.from_protein_ligand_dicts(ligand_dict=ligand_dict)


class TestComplexDataIncAndNumNodes(unittest.TestCase):
    """Tests for ComplexData index increment methods and num_nodes property."""

    def test_inc_offsets_use_expected_sizes(self) -> None:
        """Tests that __inc__ returns correct offset values based on tensor sizes for various attributes."""
        data = ComplexData(
            cpx_pos=torch.zeros((4, 3), dtype=torch.float32),
            y_pos=torch.zeros((2, 3), dtype=torch.float32),
            pos_fake=torch.zeros((3, 3), dtype=torch.float32),
            pos_real=torch.zeros((5, 3), dtype=torch.float32),
            step_batch=torch.tensor([0, 0, 1, 2], dtype=torch.long),
            ligand_element=torch.tensor([6, 8], dtype=torch.long),
            edge_query_index_0=torch.tensor([0, 1, 0], dtype=torch.long),
        )

        cpx_size = data.cpx_pos.size(0)
        self.assertEqual(data.__inc__("idx_ligand_ctx_in_cpx", torch.tensor([0])), cpx_size)
        self.assertEqual(data.__inc__("cpx_knn_edge_index", torch.tensor([0])), cpx_size)
        self.assertEqual(data.__inc__("pos_fake_knn_edge_idx_0", torch.tensor([0])), cpx_size)
        self.assertEqual(data.__inc__("pos_real_knn_edge_idx_0", torch.tensor([0])), cpx_size)

        self.assertEqual(data.__inc__("edge_query_index_0", torch.tensor([0])), data.y_pos.size(0))
        self.assertEqual(data.__inc__("pos_query_knn_edge_idx_1", torch.tensor([0])), data.y_pos.size(0))
        self.assertEqual(data.__inc__("pos_fake_knn_edge_idx_1", torch.tensor([0])), data.pos_fake.size(0))
        self.assertEqual(data.__inc__("pos_real_knn_edge_idx_1", torch.tensor([0])), data.pos_real.size(0))

        self.assertEqual(int(data.__inc__("step_batch", torch.tensor([0])).item()), 3)
        self.assertEqual(data.__inc__("ligand_bond_index", torch.tensor([[0], [1]])), torch.Size([2]))
        self.assertEqual(
            data.__inc__("index_real_cps_edge_for_atten", torch.tensor([0])),
            data.edge_query_index_0.size(0),
        )

    def test_num_nodes_depends_on_is_traj(self) -> None:
        """Tests that num_nodes changes based on the is_traj flag (protein+context vs cpx_pos size)."""
        data = ComplexData(
            protein_pos=torch.zeros((2, 3), dtype=torch.float32),
            context_idx=torch.tensor([0, 1, 2], dtype=torch.long),
            cpx_pos=torch.zeros((10, 3), dtype=torch.float32),
        )
        self.assertEqual(data.num_nodes, 5)

        data.is_traj = True
        self.assertEqual(data.num_nodes, 10)


class TestComplexDataTrajectory(unittest.TestCase):
    """Tests for ComplexDataTrajectory collation and validation helpers."""

    def test_from_steps_builds_batches(self) -> None:
        """Tests that from_steps builds step assignment vectors and offsets indices."""
        base_fields = {
            "protein_pos": torch.zeros((2, 3), dtype=torch.float32),
            "protein_atom_feature": torch.zeros((2, 2), dtype=torch.float32),
            "ligand_pos": torch.zeros((2, 3), dtype=torch.float32),
            "ligand_element": torch.tensor([6, 8], dtype=torch.long),
            "ligand_bond_index": torch.empty((2, 0), dtype=torch.long),
            "ligand_bond_type": torch.empty((0,), dtype=torch.long),
            "ligand_atom_feature_full": torch.zeros((2, 16), dtype=torch.float32),
        }
        step1 = _make_traj_step(num_context=1, num_protein=2, base_fields=base_fields, step_batch_value=7)
        step2 = _make_traj_step(num_context=2, num_protein=2, base_fields=base_fields, step_batch_value=9)

        traj = ComplexDataTrajectory.from_steps([step1, step2])

        self.assertIsInstance(traj, ComplexDataTrajectory)
        expected_step_batch = torch.cat(
            [
                torch.zeros(step1.cpx_pos.size(0), dtype=torch.long),
                torch.ones(step2.cpx_pos.size(0), dtype=torch.long),
            ]
        )
        self.assertTrue(torch.equal(traj.step_batch, expected_step_batch))
        self.assertTrue(torch.equal(traj.cpx_pos_batch, traj.step_batch))
        self.assertTrue(torch.equal(traj.y_pos_batch, torch.tensor([0, 1], dtype=torch.long)))
        self.assertTrue(torch.equal(traj.edge_label_batch, torch.tensor([0, 1], dtype=torch.long)))
        self.assertTrue(torch.equal(traj.atom_label_batch, torch.tensor([0, 1], dtype=torch.long)))
        self.assertTrue(torch.equal(traj.edge_query_index_0, torch.tensor([0, 1], dtype=torch.long)))
        self.assertTrue(
            torch.equal(
                traj.index_real_cps_edge_for_atten,
                torch.tensor([[0, 1], [0, 1]], dtype=torch.long),
            )
        )
        self.assertEqual(traj.num_steps, 2)

    def test_from_steps_rejects_mismatched_base_fields(self) -> None:
        """Tests that from_steps enforces identical base fields across steps."""
        base_fields = {
            "protein_pos": torch.zeros((2, 3), dtype=torch.float32),
            "protein_atom_feature": torch.zeros((2, 2), dtype=torch.float32),
            "ligand_pos": torch.zeros((2, 3), dtype=torch.float32),
            "ligand_element": torch.tensor([6, 8], dtype=torch.long),
            "ligand_bond_index": torch.empty((2, 0), dtype=torch.long),
            "ligand_bond_type": torch.empty((0,), dtype=torch.long),
            "ligand_atom_feature_full": torch.zeros((2, 16), dtype=torch.float32),
        }
        step1 = _make_traj_step(num_context=1, num_protein=2, base_fields=base_fields)
        mismatched = dict(base_fields)
        mismatched["ligand_element"] = torch.tensor([6, 7], dtype=torch.long)
        step2 = _make_traj_step(num_context=2, num_protein=2, base_fields=mismatched)

        with self.assertRaises(ValueError):
            ComplexDataTrajectory.from_steps([step1, step2])

    def test_validate_enforces_strict_invariants(self) -> None:
        """Tests that validate rejects dtype and batch mismatches under strict mode."""
        base_fields = {
            "protein_pos": torch.zeros((2, 3), dtype=torch.float32),
            "protein_atom_feature": torch.zeros((2, 2), dtype=torch.float32),
            "ligand_pos": torch.zeros((2, 3), dtype=torch.float32),
            "ligand_element": torch.tensor([6, 8], dtype=torch.long),
            "ligand_bond_index": torch.empty((2, 0), dtype=torch.long),
            "ligand_bond_type": torch.empty((0,), dtype=torch.long),
            "ligand_atom_feature_full": torch.zeros((2, 16), dtype=torch.float32),
        }
        step1 = _make_traj_step(num_context=1, num_protein=2, base_fields=base_fields)
        step2 = _make_traj_step(num_context=2, num_protein=2, base_fields=base_fields)
        traj = ComplexDataTrajectory.from_steps([step1, step2])

        traj.step_batch = traj.step_batch.to(torch.int32)
        with self.assertRaises(ValueError):
            traj.validate(strict=True)
        traj.validate(strict=False)

        traj.step_batch = traj.step_batch.to(torch.long)
        traj.cpx_pos_batch = traj.step_batch + 1
        with self.assertRaises(ValueError):
            traj.validate(strict=True)

        traj.step_batch = traj.step_batch.clone()
        traj.step_batch[0] = -1
        with self.assertRaises(ValueError):
            traj.validate(strict=False)


class TestCollationAndBatching(unittest.TestCase):
    """Tests for batch collation functions and data loader batching behavior."""

    def test_make_batch_collate_respects_follow_batch_and_exclude(self) -> None:
        """Tests that make_batch_collate respects follow_batch and exclude_keys parameters."""
        data1 = _basic_complex(protein_count=1, ligand_count=2, context_count=1, debug="skip")
        data2 = _basic_complex(protein_count=2, ligand_count=1, context_count=2, debug="skip")

        collate = make_batch_collate(
            follow_batch=("ligand_element",),
            exclude_keys=("debug",),
        )
        batch = collate([data1, data2])

        self.assertIsInstance(batch, Batch)
        self.assertTrue(hasattr(batch, "ligand_element_batch"))
        self.assertFalse(hasattr(batch, "protein_element_batch"))
        self.assertNotIn("debug", batch.to_dict())
        self.assertEqual(
            batch.ligand_element_batch.numel(),
            data1.ligand_element.numel() + data2.ligand_element.numel(),
        )

    def test_protein_ligand_loader_uses_default_follow_batch(self) -> None:
        """Tests that ProteinLigandDataLoader uses default follow_batch for protein and ligand elements."""
        items = [
            _basic_complex(protein_count=1, ligand_count=2, context_count=1),
            _basic_complex(protein_count=2, ligand_count=1, context_count=2),
        ]
        loader = ProteinLigandDataLoader(_ComplexDataset(items), batch_size=2, shuffle=False)
        batch = next(iter(loader))

        self.assertIsInstance(batch, Batch)
        self.assertTrue(hasattr(batch, "ligand_element_batch"))
        self.assertTrue(hasattr(batch, "protein_element_batch"))
        self.assertEqual(
            batch.protein_element_batch.numel(),
            items[0].protein_element.numel() + items[1].protein_element.numel(),
        )

    def test_batch_from_data_list_default_follow_batch(self) -> None:
        """Tests that batch_from_data_list uses default follow_batch for protein and ligand elements."""
        data_list = [
            _basic_complex(protein_count=1, ligand_count=2, context_count=1),
            _basic_complex(protein_count=2, ligand_count=1, context_count=2),
        ]

        batch = batch_from_data_list(data_list)

        self.assertIsInstance(batch, Batch)
        self.assertTrue(hasattr(batch, "ligand_element_batch"))
        self.assertTrue(hasattr(batch, "protein_element_batch"))


class TestTorchifyDict(unittest.TestCase):
    """Tests for torchify_dict conversion utility."""

    def test_torchify_dict_converts_numpy_and_keeps_objects(self) -> None:
        """Tests that torchify_dict converts numpy arrays to tensors while preserving other
        objects and maintaining references."""
        arr = np.array([1.0, 2.0], dtype=np.float32)
        sentinel: dict[str, int] = {"key": 1}
        output = torchify_dict({"arr": arr, "sentinel": sentinel})

        self.assertIsInstance(output["arr"], torch.Tensor)
        self.assertIs(output["sentinel"], sentinel)

        arr[0] = 5.0
        self.assertEqual(output["arr"][0].item(), 5.0)
