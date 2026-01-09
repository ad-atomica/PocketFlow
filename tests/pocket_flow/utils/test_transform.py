from __future__ import annotations

import importlib.util
import unittest

import torch

from pocket_flow.utils.data import ComplexData, ComplexDataTrajectory

REQUIRED_DEPS = ("torch", "torch_geometric")


def _deps_available() -> bool:
    """Check if required dependencies (torch and torch_geometric) are available."""
    return all(importlib.util.find_spec(mod) for mod in REQUIRED_DEPS)


@unittest.skipUnless(_deps_available(), "requires torch + torch_geometric")
class TestTrajCompose(unittest.TestCase):
    """Test TrajCompose transform composition on list inputs."""

    def test_list_input_chains_transforms_sequentially(self) -> None:
        """Test that transforms are applied sequentially to each list element."""
        from pocket_flow.utils.transform import TrajCompose

        def t1(x: str) -> str:
            return f"{x}A"

        def t2(x: str) -> list[str]:
            return [f"{x}B"]

        transform = TrajCompose([t1, t2])  # type: ignore[arg-type]
        out = transform(["x", "y"])  # type: ignore[arg-type]
        self.assertEqual(out, ["xAB", "yAB"])

    def test_list_input_allows_scalar_outputs(self) -> None:
        """Test that transforms can return scalar outputs that are wrapped into lists."""
        from pocket_flow.utils.transform import TrajCompose

        def t1(x: str) -> str:
            return f"{x}A"

        def t3(x: str) -> str:
            return f"{x}C"

        transform = TrajCompose([t1, t3])  # type: ignore[arg-type]
        out = transform(["x", "y"])  # type: ignore[arg-type]
        self.assertEqual(out, ["xAC", "yAC"])


def _make_ligand_chain() -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[int, list[int]]
]:
    """Create a test ligand chain with 3 atoms (C, H, O) and their bonds."""
    ligand_element = torch.tensor([6, 1, 8], dtype=torch.long)
    ligand_pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    ligand_bond_index = torch.tensor(
        [
            [0, 1, 0, 2],
            [1, 0, 2, 0],
        ],
        dtype=torch.long,
    )
    ligand_bond_type = torch.tensor([1, 1, 2, 2], dtype=torch.long)
    ligand_nbh_list = {0: [1, 2], 1: [0], 2: [0]}
    return ligand_element, ligand_pos, ligand_bond_index, ligand_bond_type, ligand_nbh_list


def _make_collate_item(*, num_context: int, num_protein: int) -> ComplexData:
    """Create a test ComplexData item with specified context and protein atom counts."""
    from pocket_flow.utils.data import ComplexData

    num_cpx = num_context + num_protein
    cpx_pos = torch.full((num_cpx, 3), float(num_cpx), dtype=torch.float32)
    cpx_feature = torch.full((num_cpx, 2), float(num_cpx), dtype=torch.float32)
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    return ComplexData(
        protein_pos=torch.zeros((num_protein, 3), dtype=torch.float32),
        protein_atom_feature=torch.zeros((num_protein, 2), dtype=torch.float32),
        ligand_pos=torch.zeros((2, 3), dtype=torch.float32),
        ligand_element=torch.tensor([6, 8], dtype=torch.long),
        ligand_bond_index=torch.empty((2, 0), dtype=torch.long),
        ligand_bond_type=torch.empty((0,), dtype=torch.long),
        ligand_atom_feature_full=torch.zeros((2, 16), dtype=torch.float32),
        cpx_pos=cpx_pos,
        cpx_feature=cpx_feature,
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
        step_batch=torch.tensor([0], dtype=torch.long),
    )


@unittest.skipUnless(_deps_available(), "requires torch + torch_geometric")
class TestRefineData(unittest.TestCase):
    """Test RefineData transform that removes hydrogens and updates bond structures."""

    def test_removes_hydrogens_and_updates_bonds(self) -> None:
        """Test that hydrogen atoms are removed from both protein and ligand,
        and bond indices are updated accordingly."""
        from pocket_flow.utils.data import ComplexData
        from pocket_flow.utils.transform import RefineData

        ligand_element, ligand_pos, ligand_bond_index, ligand_bond_type, ligand_nbh_list = (
            _make_ligand_chain()
        )
        data = ComplexData(
            protein_element=torch.tensor([6, 1, 8], dtype=torch.long),
            protein_atom_name=["C1", "H1", "O1"],
            protein_atom_to_aa_type=torch.tensor([0, 1, 2], dtype=torch.long),
            protein_is_backbone=torch.tensor([1, 0, 1], dtype=torch.bool),
            protein_pos=torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                ],
                dtype=torch.float32,
            ),
            ligand_element=ligand_element,
            ligand_pos=ligand_pos,
            ligand_atom_feature=torch.tensor(
                [
                    [1.0, 0.0],
                    [2.0, 0.0],
                    [3.0, 0.0],
                ],
                dtype=torch.float32,
            ),
            ligand_nbh_list=ligand_nbh_list,
            ligand_bond_index=ligand_bond_index,
            ligand_bond_type=ligand_bond_type,
        )

        refined = RefineData()(data)

        self.assertEqual(refined.protein_atom_name, ["C1", "O1"])
        self.assertTrue(torch.equal(refined.protein_element, torch.tensor([6, 8])))
        self.assertTrue(torch.equal(refined.protein_atom_to_aa_type, torch.tensor([0, 2])))
        self.assertTrue(torch.equal(refined.protein_is_backbone, torch.tensor([1, 1], dtype=torch.bool)))
        self.assertEqual(refined.ligand_element.tolist(), [6, 8])
        self.assertTrue(torch.equal(refined.ligand_atom_feature, torch.tensor([[1.0, 0.0], [3.0, 0.0]])))
        self.assertEqual(refined.ligand_nbh_list, {0: [1], 1: [0]})
        self.assertTrue(
            torch.equal(
                refined.ligand_bond_index,
                torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            )
        )
        self.assertTrue(torch.equal(refined.ligand_bond_type, torch.tensor([2, 2], dtype=torch.long)))


@unittest.skipUnless(_deps_available(), "requires torch + torch_geometric")
class TestLigandCountNeighbors(unittest.TestCase):
    """Test LigandCountNeighbors transform that computes neighbor and bond statistics."""

    def test_counts_neighbors_valence_and_bond_types(self) -> None:
        """Test that neighbor counts, atom valences, and bond type
        distributions are correctly computed from bond indices."""
        from pocket_flow.utils.data import ComplexData
        from pocket_flow.utils.transform import LigandCountNeighbors

        bond_index = torch.tensor(
            [
                [0, 1, 1, 2],
                [1, 0, 2, 1],
            ],
            dtype=torch.long,
        )
        bond_type = torch.tensor([1, 1, 2, 2], dtype=torch.long)
        data = ComplexData(
            ligand_element=torch.tensor([6, 7, 8], dtype=torch.long),
            ligand_bond_index=bond_index,
            ligand_bond_type=bond_type,
        )

        out = LigandCountNeighbors()(data)

        self.assertTrue(torch.equal(out.ligand_num_neighbors, torch.tensor([1, 2, 1], dtype=torch.long)))
        self.assertTrue(torch.equal(out.ligand_atom_valence, torch.tensor([1, 3, 2], dtype=torch.long)))
        self.assertTrue(
            torch.equal(
                out.ligand_atom_num_bonds,
                torch.tensor(
                    [
                        [1, 0, 0],
                        [1, 1, 0],
                        [0, 1, 0],
                    ],
                    dtype=torch.long,
                ),
            )
        )


@unittest.skipUnless(_deps_available(), "requires torch + torch_geometric")
class TestFeaturizers(unittest.TestCase):
    """Test atom featurization transforms for protein and ligand atoms."""

    def test_featurize_protein_atom_builds_expected_layout(self) -> None:
        """Test that protein atom features have correct dimensions and
        encode element, amino acid type, and backbone status."""
        from pocket_flow.utils.data import ComplexData
        from pocket_flow.utils.transform import FeaturizeProteinAtom

        data = ComplexData(
            protein_element=torch.tensor([6, 1, 34], dtype=torch.long),
            protein_atom_to_aa_type=torch.tensor([0, 5, 10], dtype=torch.long),
            protein_is_backbone=torch.tensor([1, 0, 1], dtype=torch.bool),
        )

        out = FeaturizeProteinAtom()(data)
        feature = out.protein_atom_feature

        self.assertEqual(feature.shape[0], 3)
        self.assertEqual(feature.shape[1], FeaturizeProteinAtom().feature_dim)
        self.assertEqual(int(feature[0, 0].item()), 1)
        self.assertTrue(torch.equal(feature[:, -1], torch.zeros(3, dtype=torch.long)))
        self.assertEqual(int(feature[1, :5].sum().item()), 0)

    def test_featurize_ligand_atom_uses_counts(self) -> None:
        """Test that ligand atom features incorporate neighbor counts,
        valence, and bond type distributions."""
        from pocket_flow.utils.data import ComplexData
        from pocket_flow.utils.transform import FeaturizeLigandAtom

        data = ComplexData(
            ligand_element=torch.tensor([6, 8], dtype=torch.long),
            ligand_num_neighbors=torch.tensor([1, 2], dtype=torch.long),
            ligand_atom_valence=torch.tensor([1, 3], dtype=torch.long),
            ligand_atom_num_bonds=torch.tensor([[1, 0, 0], [1, 1, 0]], dtype=torch.long),
        )

        out = FeaturizeLigandAtom()(data)
        feature = out.ligand_atom_feature_full

        self.assertEqual(feature.shape, (2, 16))
        self.assertTrue(torch.equal(feature[:, 10], torch.ones(2, dtype=torch.long)))
        self.assertTrue(torch.equal(feature[:, 11], torch.tensor([1, 2], dtype=torch.long)))
        self.assertTrue(torch.equal(feature[:, 12], torch.tensor([1, 3], dtype=torch.long)))
        self.assertTrue(
            torch.equal(
                feature[:, 13:16],
                torch.tensor([[1, 0, 0], [1, 1, 0]], dtype=torch.long),
            )
        )


@unittest.skipUnless(_deps_available(), "requires torch + torch_geometric")
class TestLigandTrajectory(unittest.TestCase):
    """Test LigandTrajectory transform that generates stepwise generation trajectories."""

    def test_builds_stepwise_context_and_masked_sets(self) -> None:
        """Test that trajectory steps correctly partition atoms into
        context (revealed) and masked (to be generated) sets."""
        from pocket_flow.utils.data import ComplexData
        from pocket_flow.utils.transform import FeaturizeLigandAtom, LigandCountNeighbors, LigandTrajectory

        ligand_element, ligand_pos, ligand_bond_index, ligand_bond_type, ligand_nbh_list = (
            _make_ligand_chain()
        )
        data = ComplexData(
            ligand_element=ligand_element,
            ligand_pos=ligand_pos,
            ligand_bond_index=ligand_bond_index,
            ligand_bond_type=ligand_bond_type,
            ligand_nbh_list=ligand_nbh_list,
        )
        data = LigandCountNeighbors()(data)
        data = FeaturizeLigandAtom()(data)

        traj = LigandTrajectory(y_pos_std=0.0)(data)

        self.assertEqual(len(traj), ligand_element.numel())
        for step_idx, step in enumerate(traj):
            self.assertEqual(step.context_idx.numel(), step_idx)
            self.assertEqual(step.masked_idx.numel(), ligand_element.numel() - step_idx)
            self.assertEqual(step.ligand_context_pos.size(0), step.context_idx.numel())
            self.assertEqual(step.ligand_masked_pos.size(0), step.masked_idx.numel())


@unittest.skipUnless(_deps_available(), "requires torch + torch_geometric")
class TestComplexDataTrajectoryFromSteps(unittest.TestCase):
    """Test ComplexDataTrajectory.from_steps with proper index offsetting."""

    def test_offsets_indices_across_items(self) -> None:
        """Test that indices for edges, positions, and other graph structures are
        correctly offset when batching multiple items."""
        item1 = _make_collate_item(num_context=1, num_protein=2)
        item2 = _make_collate_item(num_context=2, num_protein=2)

        batch = ComplexDataTrajectory.from_steps([item1, item2])

        self.assertIsInstance(batch, ComplexDataTrajectory)
        self.assertTrue(torch.equal(batch.idx_ligand_ctx_in_cpx, torch.tensor([0, 3, 4], dtype=torch.long)))
        self.assertTrue(torch.equal(batch.idx_protein_in_cpx, torch.tensor([1, 2, 5, 6], dtype=torch.long)))
        self.assertTrue(torch.equal(batch.edge_query_index_0, torch.tensor([0, 1], dtype=torch.long)))
        self.assertTrue(torch.equal(batch.edge_query_index_1, torch.tensor([0, 3], dtype=torch.long)))
        self.assertTrue(torch.equal(batch.pos_fake_knn_edge_idx_1, torch.tensor([0, 1], dtype=torch.long)))
        self.assertTrue(torch.equal(batch.pos_real_knn_edge_idx_1, torch.tensor([0, 1], dtype=torch.long)))
        self.assertTrue(
            torch.equal(
                batch.tri_edge_index,
                torch.tensor([[0, 3], [1, 4]], dtype=torch.long),
            )
        )
        self.assertTrue(
            torch.equal(
                batch.index_real_cps_edge_for_atten,
                torch.tensor([[0, 1], [0, 1]], dtype=torch.long),
            )
        )
        self.assertTrue(
            torch.equal(
                batch.step_batch,
                torch.tensor([0, 0, 0, 1, 1, 1, 1], dtype=torch.long),
            )
        )
        self.assertTrue(torch.equal(batch.cpx_pos_batch, batch.step_batch))
        self.assertTrue(torch.equal(batch.y_pos_batch, torch.tensor([0, 1], dtype=torch.long)))
        self.assertTrue(torch.equal(batch.edge_label_batch, torch.tensor([0, 1], dtype=torch.long)))
        self.assertTrue(torch.equal(batch.atom_label_batch, torch.tensor([0, 1], dtype=torch.long)))
        self.assertEqual(batch.cpx_pos.size(0), 7)
        self.assertEqual(batch.cpx_pos_batch.numel(), 7)

    def test_rejects_mismatched_base_fields(self) -> None:
        """Test that from_steps fails fast if steps disagree on shared base fields."""

        item1 = _make_collate_item(num_context=1, num_protein=2)
        item2 = _make_collate_item(num_context=2, num_protein=2)
        item2.ligand_element = torch.tensor([6, 7], dtype=torch.long)

        with self.assertRaises(ValueError):
            ComplexDataTrajectory.from_steps([item1, item2])


@unittest.skipUnless(_deps_available(), "requires torch + torch_geometric")
class TestEdgeIndexMapping(unittest.TestCase):
    """Tests for ctx/cpx index mapping in transform helpers."""

    def test_sample_edge_with_radius_maps_ctx_to_cpx(self) -> None:
        """Test sample_edge_with_radius maps ctx indices to cpx indices."""
        from pocket_flow.utils.neighbor_search import radius
        from pocket_flow.utils.transform_utils import sample_edge_with_radius

        data = ComplexData(
            y_pos=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
            ligand_context_pos=torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                ],
                dtype=torch.float32,
            ),
            context_idx=torch.tensor([0, 1], dtype=torch.long),
            masked_idx=torch.tensor([0], dtype=torch.long),
            ligand_bond_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            ligand_bond_type=torch.tensor([1, 1], dtype=torch.long),
            idx_ligand_ctx_in_cpx=torch.tensor([2, 0], dtype=torch.long),
        )

        edge_index_radius = radius(
            data.ligand_context_pos,
            data.y_pos,
            r=4.0,
            num_workers=1,
        )
        expected = data.idx_ligand_ctx_in_cpx[edge_index_radius[1]]

        out = sample_edge_with_radius(data, r=4.0, num_workers=1)

        self.assertTrue(torch.equal(out.edge_query_index_1, expected))

    def test_get_tri_edges_maps_to_cpx_indices(self) -> None:
        """Test get_tri_edges maps ctx triangle endpoints into cpx space."""
        from pocket_flow.utils.transform_utils import get_tri_edges

        edge_index_query_ctx = torch.tensor([[0, 0], [0, 1]], dtype=torch.long)
        pos_query = torch.zeros((1, 3), dtype=torch.float32)
        idx_ligand_ctx_in_cpx = torch.tensor([2, 0], dtype=torch.long)
        ligand_bond_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        ligand_bond_type = torch.tensor([1, 1], dtype=torch.long)

        _, tri_edge_index, tri_edge_feat = get_tri_edges(
            edge_index_query_ctx,
            pos_query,
            idx_ligand_ctx_in_cpx,
            ligand_bond_index,
            ligand_bond_type,
        )

        expected = torch.tensor([[2, 0, 2, 0], [2, 2, 0, 0]], dtype=torch.long)
        self.assertTrue(torch.equal(tri_edge_index, expected))
        self.assertEqual(int(tri_edge_feat[1, 2].item()), 1)
