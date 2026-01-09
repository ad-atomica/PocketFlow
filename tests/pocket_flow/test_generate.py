from __future__ import annotations

import ast
import importlib.util
import tempfile
import types
import unittest
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    import torch
    from rdkit.Chem.rdchem import Mol

    from pocket_flow.gdbp_model.pocket_flow import PocketFlow
    from pocket_flow.generate import Generate
    from pocket_flow.utils.data import ComplexData


# =============================================================================
# Constants
# =============================================================================

HIDDEN_CHANNELS = 8
KNN = 4
DEFAULT_BOND_LENGTH_RANGE = (1.0, 2.0)
DEFAULT_MIN_DIST_INTER_MOL = 3.0
REQUIRED_DEPS = ("torch", "rdkit", "torch_geometric")


# =============================================================================
# Utilities
# =============================================================================


def _deps_available() -> bool:
    """Check if all required dependencies are installed."""
    return all(importlib.util.find_spec(mod) for mod in REQUIRED_DEPS)


def _make_model_config() -> types.SimpleNamespace:
    """Create a minimal model config namespace for testing."""
    return types.SimpleNamespace(
        hidden_channels=HIDDEN_CHANNELS,
        encoder=types.SimpleNamespace(knn=KNN),
    )


def _make_mock_model() -> PocketFlow:
    """Create a mock model namespace cast to PocketFlow for testing."""
    model = types.SimpleNamespace(config=_make_model_config())
    return cast("PocketFlow", model)


def _make_mock_data(**attrs: Any) -> ComplexData:
    """Create a mock data object cast to ComplexData for testing.

    Args:
        **attrs: Attributes to set on the mock data object.

    Returns:
        A SimpleNamespace cast to ComplexData with the given attributes.
    """
    data = types.SimpleNamespace(**attrs)
    # Add required methods for ComplexData interface
    data.__contains__ = lambda key: hasattr(data, key)
    data.__getitem__ = lambda key: getattr(data, key)
    return cast("ComplexData", data)


def _make_cloneable_data() -> ComplexData:
    """Create a mock cloneable data object for generate() tests.

    Returns:
        A SimpleNamespace cast to ComplexData with clone/detach/to methods.
    """
    data = types.SimpleNamespace()
    data.clone = lambda: data
    data.detach = lambda: data
    data.to = lambda _device: data
    return cast("ComplexData", data)


# =============================================================================
# Test Classes
# =============================================================================


class TestGenerateModule(unittest.TestCase):
    """Tests for the generate module structure."""

    def test_module_defines_Generate_class(self) -> None:
        """Test that the generate.py module defines a Generate class by parsing its AST."""
        root = Path(__file__).resolve().parents[2]
        path = root / "pocket_flow" / "generate.py"
        module = ast.parse(path.read_text(encoding="utf-8"))
        class_names = {n.name for n in module.body if isinstance(n, ast.ClassDef)}
        self.assertIn("Generate", class_names, "Generate class not found in module")


@unittest.skipUnless(_deps_available(), "requires torch + rdkit + torch_geometric")
class TestGenerateHelpers(unittest.TestCase):
    """Tests for Generate class helper methods and initialization."""

    # -------------------------------------------------------------------------
    # Setup helpers
    # -------------------------------------------------------------------------

    @classmethod
    def setUpClass(cls) -> None:
        """Import dependencies once for the entire test class."""
        import torch as torch_module
        from rdkit import Chem as chem_module

        from pocket_flow.generate import Generate as generate_class

        cls.torch = torch_module
        cls.Chem = chem_module
        cls.Generate = generate_class

    def _make_generator(
        self,
        *,
        focal_logits: torch.Tensor | None,
        pos_outputs: tuple[torch.Tensor, ...] | None,
        min_dist_inter_mol: float = DEFAULT_MIN_DIST_INTER_MOL,
        bond_length_range: tuple[float, float] = DEFAULT_BOND_LENGTH_RANGE,
    ) -> Generate:
        """Create a Generate instance with controlled focal/pos predictors.

        Args:
            focal_logits: Fixed logits for focal network, or None to skip.
            pos_outputs: Fixed outputs for position predictor, or None to skip.
            min_dist_inter_mol: Minimum inter-molecular distance.
            bond_length_range: Valid bond length range (min, max).

        Returns:
            Configured Generate instance for testing.
        """

        class FixedFocalNet:
            """Mock focal network returning pre-configured logits."""

            def __init__(self, logits: torch.Tensor) -> None:
                self._logits = logits

            def __call__(self, h_ctx: tuple[Any, ...], ctx_idx: torch.Tensor) -> torch.Tensor:
                return self._logits.to(ctx_idx.device)

        class FixedPosPredictor:
            """Mock position predictor returning pre-configured outputs."""

            def __init__(self, outputs: tuple[torch.Tensor, ...]) -> None:
                self._outputs = outputs

            def __call__(
                self,
                h_cpx: list[Any],
                focal_idx: torch.Tensor,
                cpx_pos: torch.Tensor,
                atom_type_emb: torch.Tensor | None = None,
            ) -> tuple[torch.Tensor, ...]:
                return self._outputs

        model = types.SimpleNamespace(config=_make_model_config())
        if focal_logits is not None:
            model.focal_net = FixedFocalNet(focal_logits)
        if pos_outputs is not None:
            model.pos_predictor = FixedPosPredictor(pos_outputs)

        return self.Generate(
            model=cast("PocketFlow", model),
            transform=lambda data: data,
            device="cpu",
            min_dist_inter_mol=min_dist_inter_mol,
            bond_length_range=bond_length_range,
            num_workers=1,
        )

    def _make_h_ctx(self, size: int, hidden: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        """Create a mock h_ctx tuple for testing."""
        torch = self.torch
        return (torch.zeros(size, hidden), torch.zeros(size, hidden, 3))

    # -------------------------------------------------------------------------
    # Initialization tests
    # -------------------------------------------------------------------------

    def test_init_rejects_bad_temperature(self) -> None:
        """Test that Generate.__init__ raises ValueError for invalid temperature parameter."""
        model = _make_mock_model()

        with self.assertRaises(ValueError, msg="Should reject temperature with wrong length"):
            self.Generate(
                model=model,
                transform=lambda data: data,
                temperature=(1.0,),  # type: ignore[arg-type]
                device="cpu",
                num_workers=1,
            )

    def test_init_rejects_invalid_bond_length_range(self) -> None:
        """Test that Generate.__init__ raises ValueError for invalid bond length range where min > max."""
        model = _make_mock_model()

        with self.assertRaises(ValueError, msg="Should reject bond_length_range with min > max"):
            self.Generate(
                model=model,
                transform=lambda data: data,
                bond_length_range=(2.0, 1.0),
                device="cpu",
                num_workers=1,
            )

    def test_init_rejects_equal_bond_length_range(self) -> None:
        """Test that Generate.__init__ raises ValueError when min == max in bond length range."""
        model = _make_mock_model()

        with self.assertRaises(ValueError, msg="Should reject bond_length_range with min == max"):
            self.Generate(
                model=model,
                transform=lambda data: data,
                bond_length_range=(1.5, 1.5),
                device="cpu",
                num_workers=1,
            )

    # -------------------------------------------------------------------------
    # _safe_multinomial tests
    # -------------------------------------------------------------------------

    def test_safe_multinomial_fallback_false(self) -> None:
        """Test _safe_multinomial returns False for invalid weights with 'false' fallback."""
        torch = self.torch
        weights = torch.tensor([float("nan"), float("inf"), -1.0])

        result = self.Generate._safe_multinomial(weights, fallback="false")

        self.assertIs(result, False, "Should return False for invalid weights")

    def test_safe_multinomial_fallback_argmax(self) -> None:
        """Test _safe_multinomial returns argmax index for invalid weights with 'argmax' fallback."""
        torch = self.torch
        weights = torch.tensor([float("nan"), float("inf"), -1.0])

        result = self.Generate._safe_multinomial(weights, fallback="argmax")

        self.assertEqual(result.numel(), 1, "Should return single index")
        self.assertEqual(int(result.item()), 0, "Argmax should select index 0")

    def test_safe_multinomial_fallback_uniform(self) -> None:
        """Test _safe_multinomial returns valid random index with 'uniform' fallback."""
        torch = self.torch
        weights = torch.tensor([float("nan"), float("inf"), -1.0])

        result = self.Generate._safe_multinomial(weights, fallback="uniform")

        self.assertEqual(result.numel(), 1, "Should return single index")
        self.assertIn(int(result.item()), range(weights.numel()), "Index should be in valid range")

    def test_safe_multinomial_empty_weights(self) -> None:
        """Test _safe_multinomial handles empty weight tensor."""
        torch = self.torch
        weights = torch.tensor([])

        result_false = self.Generate._safe_multinomial(weights, fallback="false")
        result_uniform = self.Generate._safe_multinomial(weights, fallback="uniform")

        self.assertIs(result_false, False)
        self.assertIs(result_uniform, False)

    def test_safe_multinomial_single_element(self) -> None:
        """Test _safe_multinomial with single element always returns index 0."""
        torch = self.torch
        weights = torch.tensor([5.0])

        result = self.Generate._safe_multinomial(weights, fallback="uniform")

        self.assertEqual(int(result.item()), 0)

    def test_safe_multinomial_valid_weights(self) -> None:
        """Test _safe_multinomial with valid weights performs multinomial sampling."""
        torch = self.torch
        # Heavily biased weights - index 2 should almost always be selected
        weights = torch.tensor([0.0, 0.0, 1.0])

        result = self.Generate._safe_multinomial(weights, fallback="uniform")

        self.assertEqual(result.numel(), 1)
        self.assertEqual(int(result.item()), 2, "Should select index with all weight")

    # -------------------------------------------------------------------------
    # _choose_focal tests
    # -------------------------------------------------------------------------

    def test_choose_focal_uses_surface_mask_threshold(self) -> None:
        """Test that _choose_focal respects surface mask and focus threshold for focal atom selection."""
        torch = self.torch
        ctx_idx = torch.tensor([10, 11, 12], dtype=torch.long)
        logits = torch.tensor([0.0, -2.0, 1.0])
        surf_mask = torch.tensor([True, False, True])

        out = self.Generate._choose_focal(
            focal_net=lambda h_ctx, idx: logits,
            h_ctx=self._make_h_ctx(1),
            ctx_idx=ctx_idx,
            focus_threshold=0.6,
            choose_max=False,
            surf_mask=surf_mask,
        )

        self.assertIsNot(out, False, "Should return focal candidates")
        assert out is not False  # Type narrowing for mypy/pyright
        focal_idx, focal_prob = out
        self.assertTrue(
            torch.equal(focal_idx, torch.tensor([12])),
            f"Expected focal_idx [12], got {focal_idx}",
        )
        self.assertGreater(float(focal_prob.item()), 0.6)

    def test_choose_focal_returns_false_when_no_candidates(self) -> None:
        """Test that _choose_focal returns False when no candidates meet the focus threshold."""
        torch = self.torch
        ctx_idx = torch.tensor([1, 2], dtype=torch.long)
        logits = torch.tensor([-4.0, -5.0])

        out = self.Generate._choose_focal(
            focal_net=lambda h_ctx, idx: logits,
            h_ctx=self._make_h_ctx(1),
            ctx_idx=ctx_idx,
            focus_threshold=0.9,
            choose_max=False,
            surf_mask=None,
        )

        self.assertIs(out, False, "Should return False when no candidates meet threshold")

    def test_choose_focal_choose_max_selects_highest(self) -> None:
        """Test that _choose_focal with choose_max=True selects the highest scoring candidate."""
        torch = self.torch
        ctx_idx = torch.tensor([0, 1, 2], dtype=torch.long)
        logits = torch.tensor([0.5, 2.0, 1.0])  # Index 1 has highest logit

        out = self.Generate._choose_focal(
            focal_net=lambda h_ctx, idx: logits,
            h_ctx=self._make_h_ctx(1),
            ctx_idx=ctx_idx,
            focus_threshold=0.5,
            choose_max=True,
            surf_mask=None,
        )

        self.assertIsNot(out, False)
        assert out is not False  # Type narrowing for mypy/pyright
        focal_idx, _ = out
        self.assertEqual(int(focal_idx.item()), 1, "Should select index with highest logit")

    def test_choose_focal_valence_gate_blocks_candidates(self) -> None:
        """Test that choose_focal blocks candidates when all available atoms have reached maximum valence."""
        torch = self.torch
        logits = torch.tensor([2.0, 1.0])
        generator = self._make_generator(focal_logits=logits, pos_outputs=None)
        data = _make_mock_data(
            ligand_context_element=torch.zeros(4, dtype=torch.long),
            ligand_context_pos=torch.zeros(4, 3, dtype=torch.float32),
            cpx_pos=torch.zeros(4, 3, dtype=torch.float32),
            idx_ligand_ctx_in_cpx=torch.arange(4, dtype=torch.long),
            max_atom_valence=torch.ones(4, dtype=torch.long),
            ligand_context_valence=torch.ones(4, dtype=torch.long),
        )

        out = generator.choose_focal(
            h_cpx=self._make_h_ctx(2),
            cpx_index=torch.tensor([0, 1], dtype=torch.long),
            idx_ligand_ctx_in_cpx=torch.tensor([0, 1], dtype=torch.long),
            data=data,
            atom_idx=1,
        )

        self.assertIs(out, False, "Should return False when all candidates at max valence")

    def test_choose_focal_valence_gate_maps_cpx_to_ctx(self) -> None:
        """Test that valence gating maps cpx indices back to ctx indices before indexing ctx tensors."""
        torch = self.torch
        logits = torch.tensor([2.0, 1.0, 0.0, -1.0])
        generator = self._make_generator(focal_logits=logits, pos_outputs=None)
        idx_ligand_ctx_in_cpx = torch.tensor([2, 0, 3, 1], dtype=torch.long)
        data = _make_mock_data(
            ligand_context_element=torch.zeros(4, dtype=torch.long),
            ligand_context_pos=torch.zeros(4, 3, dtype=torch.float32),
            cpx_pos=torch.zeros(4, 3, dtype=torch.float32),
            idx_ligand_ctx_in_cpx=idx_ligand_ctx_in_cpx,
            max_atom_valence=torch.tensor([2, 1, 1, 1], dtype=torch.long),
            ligand_context_valence=torch.tensor([1, 1, 1, 1], dtype=torch.long),
        )

        out = generator.choose_focal(
            h_cpx=self._make_h_ctx(4),
            cpx_index=torch.tensor([0, 1], dtype=torch.long),
            idx_ligand_ctx_in_cpx=idx_ligand_ctx_in_cpx,
            data=data,
            atom_idx=1,
        )

        self.assertIsNot(out, False, "Expected mapping-aware valence gating to keep ctx0 candidate")
        assert out is not False
        focal_idx, _ = out
        self.assertTrue(torch.equal(focal_idx, torch.tensor([2], dtype=torch.long)))

    def test_assert_ctx_cpx_mapping_rejects_duplicates(self) -> None:
        """Test that duplicate ctx->cpx mappings fail fast."""
        torch = self.torch
        data = _make_mock_data(
            ligand_context_pos=torch.zeros(2, 3, dtype=torch.float32),
            cpx_pos=torch.zeros(3, 3, dtype=torch.float32),
            idx_ligand_ctx_in_cpx=torch.tensor([1, 1], dtype=torch.long),
        )

        with self.assertRaises(AssertionError):
            self.Generate._assert_ctx_cpx_mapping(data)

    # -------------------------------------------------------------------------
    # pos_generate tests
    # -------------------------------------------------------------------------

    def test_pos_generate_falls_back_to_farthest_for_first_atom(self) -> None:
        """Test that pos_generate falls back to the farthest position when generating the first atom."""
        torch = self.torch
        rel = torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        abs_pos = torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        outputs = (rel, abs_pos, torch.zeros_like(rel), torch.tensor([0.3, 0.3, 0.4]))
        generator = self._make_generator(
            focal_logits=None,
            pos_outputs=outputs,
            min_dist_inter_mol=10.0,  # All candidates fail threshold
        )

        out = generator.pos_generate(
            h_cpx=self._make_h_ctx(1),
            atom_type_emb=torch.zeros(1, HIDDEN_CHANNELS),
            focal_idx=torch.tensor([0]),
            cpx_pos=torch.zeros(1, 3),
            atom_idx=0,
        )

        self.assertIsNot(out, False)
        assert out is not False  # Type narrowing for mypy/pyright
        expected = torch.tensor([[3.0, 0.0, 0.0]])
        self.assertTrue(
            torch.equal(out, expected),
            f"Should fall back to farthest position. Got {out}, expected {expected}",
        )

    def test_pos_generate_rejects_out_of_range_candidates(self) -> None:
        """Test that pos_generate rejects position candidates outside the configured bond length range."""
        torch = self.torch
        rel = torch.tensor([[3.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
        abs_pos = torch.tensor([[3.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
        outputs = (rel, abs_pos, torch.zeros_like(rel), torch.tensor([0.5, 0.5]))
        generator = self._make_generator(
            focal_logits=None,
            pos_outputs=outputs,
            bond_length_range=(1.0, 2.0),
        )

        out = generator.pos_generate(
            h_cpx=self._make_h_ctx(1),
            atom_type_emb=torch.zeros(1, HIDDEN_CHANNELS),
            focal_idx=torch.tensor([0]),
            cpx_pos=torch.zeros(1, 3),
            atom_idx=1,
        )

        self.assertIs(out, False, "Should reject candidates outside bond length range")

    def test_pos_generate_uses_pi_for_in_range_candidates(self) -> None:
        """Test that pos_generate uses probability-weighted selection for in-range bond length candidates."""
        torch = self.torch
        rel = torch.tensor([[1.3, 0.0, 0.0], [1.4, 0.0, 0.0]])
        abs_pos = torch.tensor([[9.0, 0.0, 0.0], [8.0, 0.0, 0.0]])
        # First candidate has all the probability weight
        outputs = (rel, abs_pos, torch.zeros_like(rel), torch.tensor([1.0, 0.0]))
        generator = self._make_generator(
            focal_logits=None,
            pos_outputs=outputs,
            bond_length_range=(1.0, 2.0),
        )

        out = generator.pos_generate(
            h_cpx=self._make_h_ctx(1),
            atom_type_emb=torch.zeros(1, HIDDEN_CHANNELS),
            focal_idx=torch.tensor([0]),
            cpx_pos=torch.zeros(1, 3),
            atom_idx=1,
        )

        self.assertIsNot(out, False)
        assert out is not False  # Type narrowing for mypy/pyright
        expected = torch.tensor([[9.0, 0.0, 0.0]])
        self.assertTrue(
            torch.equal(out, expected),
            f"Should select position with highest probability. Got {out}, expected {expected}",
        )

    def test_pos_generate_boundary_bond_lengths(self) -> None:
        """Test pos_generate accepts candidates exactly at bond length boundaries (exclusive upper)."""
        torch = self.torch
        # Distance exactly at lower bound (1.0) should be excluded (> not >=)
        # Distance exactly at upper bound (2.0) should be excluded (< not <=)
        rel = torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
        abs_pos = torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
        outputs = (rel, abs_pos, torch.zeros_like(rel), torch.tensor([0.0, 0.0, 1.0]))
        generator = self._make_generator(
            focal_logits=None,
            pos_outputs=outputs,
            bond_length_range=(1.0, 2.0),
        )

        out = generator.pos_generate(
            h_cpx=self._make_h_ctx(1),
            atom_type_emb=torch.zeros(1, HIDDEN_CHANNELS),
            focal_idx=torch.tensor([0]),
            cpx_pos=torch.zeros(1, 3),
            atom_idx=1,
        )

        self.assertIsNot(out, False)
        assert out is not False  # Type narrowing for mypy/pyright
        expected = torch.tensor([[1.5, 0.0, 0.0]])
        self.assertTrue(
            torch.equal(out, expected),
            "Only middle candidate (1.5) should be in range (1.0, 2.0)",
        )

    # -------------------------------------------------------------------------
    # bond_generate tests
    # -------------------------------------------------------------------------

    def test_bond_generate_initial_step_has_no_edges(self) -> None:
        """Test that bond_generate produces no edges when generating the first atom with an empty molecule."""
        torch = self.torch
        Chem = self.Chem

        generator = self._make_generator(focal_logits=None, pos_outputs=None)
        rw_mol = Chem.RWMol()
        rw_mol.AddAtom(Chem.Atom(6))  # Carbon

        out = generator.bond_generate(
            h_cpx=self._make_h_ctx(1),
            data=_make_mock_data(),
            new_pos_to_add=torch.zeros(1, 3),
            atom_type_emb=torch.zeros(1, HIDDEN_CHANNELS),
            atom_idx=0,
            rw_mol=rw_mol,
        )

        self.assertIsNot(out, False, "First atom bond generation should succeed")
        assert out is not False  # Type narrowing for mypy/pyright
        rw_mol_out, edge_idx, bond_type = out
        self.assertEqual(rw_mol_out.GetNumAtoms(), 1, "Molecule should have 1 atom")
        self.assertEqual(tuple(edge_idx.shape), (2, 0), "Should have no edges for first atom")
        self.assertEqual(tuple(bond_type.shape), (0,), "Should have no bond types for first atom")

    def test_bond_generate_fails_fast_without_radius_neighbors(self) -> None:
        """Test that bond_generate fails immediately when no radius neighbors exist."""
        torch = self.torch
        Chem = self.Chem

        generator = self._make_generator(focal_logits=None, pos_outputs=None)
        generator.resample_edge_failed = False

        rw_mol = Chem.RWMol()
        rw_mol.AddAtom(Chem.Atom(6))
        rw_mol.AddAtom(Chem.Atom(6))

        data = _make_mock_data(
            ligand_context_pos=torch.tensor([[10.0, 0.0, 0.0]], dtype=torch.float32),
            ligand_context_bond_index=torch.empty((2, 0), dtype=torch.long),
            ligand_context_bond_type=torch.empty((0,), dtype=torch.long),
            idx_ligand_ctx_in_cpx=torch.tensor([0], dtype=torch.long),
            cpx_pos=torch.tensor([[10.0, 0.0, 0.0]], dtype=torch.float32),
        )

        out = generator.bond_generate(
            h_cpx=self._make_h_ctx(1),
            data=data,
            new_pos_to_add=torch.zeros(1, 3),
            atom_type_emb=torch.zeros(1, HIDDEN_CHANNELS),
            atom_idx=1,
            rw_mol=rw_mol,
        )

        self.assertIs(out, False, "Expected bond generation to fail without radius neighbors")
        self.assertTrue(generator.resample_edge_failed, "Failure should mark resample_edge_failed")
        self.assertEqual(rw_mol.GetNumAtoms(), 1, "Should remove the failed atom")

    def test_bond_generate_uses_cpx_positions_for_distance(self) -> None:
        """Test that bond_generate uses ctx->cpx mapping when checking bond distances."""
        torch = self.torch
        Chem = self.Chem
        from torch.distributions import Normal

        generator = self._make_generator(
            focal_logits=None,
            pos_outputs=None,
            bond_length_range=(0.5, 2.0),
        )

        class FixedEdgeFlow:
            def __init__(self, num_bond_type: int) -> None:
                self._num_bond_type = num_bond_type

            def reverse(
                self,
                *,
                edge_latent: torch.Tensor,
                pos_query: torch.Tensor,
                edge_index_query: torch.Tensor,
                cpx_pos: torch.Tensor,
                node_attr_compose: tuple[torch.Tensor, torch.Tensor],
                edge_index_q_cps_knn: torch.Tensor,
                index_real_cps_edge_for_atten: tuple[torch.Tensor, torch.Tensor],
                tri_edge_index: torch.Tensor,
                tri_edge_feat: torch.Tensor,
                atom_type_emb: torch.Tensor,
                annealing: bool = False,
            ) -> torch.Tensor:
                logits = torch.zeros(
                    (edge_index_query.size(1), self._num_bond_type), device=edge_index_query.device
                )
                logits[:, 1] = 1.0
                return logits

        generator.model.edge_flow = FixedEdgeFlow(generator.num_bond_type)
        generator.prior_edge = Normal(
            torch.zeros(generator.num_bond_type),
            torch.ones(generator.num_bond_type),
        )

        rw_mol = Chem.RWMol()
        rw_mol.AddAtom(Chem.Atom(6))
        rw_mol.AddAtom(Chem.Atom(6))

        data = _make_mock_data(
            ligand_context_pos=torch.tensor([[1.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=torch.float32),
            ligand_context_bond_index=torch.empty((2, 0), dtype=torch.long),
            ligand_context_bond_type=torch.empty((0,), dtype=torch.long),
            idx_ligand_ctx_in_cpx=torch.tensor([2, 0], dtype=torch.long),
            cpx_pos=torch.tensor(
                [
                    [100.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [5.0, 0.0, 0.0],
                ],
                dtype=torch.float32,
            ),
        )

        out = generator.bond_generate(
            h_cpx=self._make_h_ctx(4),
            data=data,
            new_pos_to_add=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
            atom_type_emb=torch.zeros(1, HIDDEN_CHANNELS),
            atom_idx=1,
            rw_mol=rw_mol,
        )

        self.assertIsNot(out, False, "Expected bond generation to succeed with mapped cpx positions")
        assert out is not False
        rw_mol_out, edge_idx, bond_type = out
        self.assertEqual(rw_mol_out.GetNumBonds(), 1)
        self.assertTrue(torch.equal(edge_idx[1], torch.tensor([0], dtype=torch.long)))
        self.assertTrue(torch.equal(bond_type, torch.tensor([1], dtype=torch.long)))


@unittest.skipUnless(_deps_available(), "requires torch + rdkit + torch_geometric")
class TestGenerateOutput(unittest.TestCase):
    """Tests for Generate class output writing and metrics calculation."""

    @classmethod
    def setUpClass(cls) -> None:
        """Import dependencies once for the entire test class."""
        import torch as torch_module
        from rdkit import Chem as chem_module

        from pocket_flow.generate import Generate as generate_class

        cls.torch = torch_module
        cls.Chem = chem_module
        cls.Generate = generate_class

    def _make_dummy_generator(self, outputs: list[tuple[Mol, Mol] | None]) -> Generate:
        """Create a DummyGenerate instance with pre-configured outputs.

        Args:
            outputs: List of (modified_mol, raw_mol) tuples or None for each
                generation attempt.

        Returns:
            DummyGenerate instance that returns outputs in order.
        """
        model = _make_mock_model()

        class DummyGenerate(self.Generate):
            """Generate subclass that returns pre-configured outputs."""

            def __init__(self, outputs: list[tuple[Mol, Mol] | None]) -> None:
                super().__init__(
                    model=model,
                    transform=lambda data: data,
                    device="cpu",
                    num_workers=1,
                )
                self._outputs = list(outputs)

            def run(self, data: Any) -> tuple[Mol, Mol] | None:  # type: ignore[override]
                if not self._outputs:
                    return None
                return self._outputs.pop(0)

        return DummyGenerate(outputs)

    def test_generate_writes_outputs_and_metrics(self) -> None:
        """Test that generate() writes SDF, SMI files and calculates validity/uniqueness metrics correctly."""
        Chem = self.Chem

        mol_a1 = Chem.MolFromSmiles("CC")
        mol_a2 = Chem.MolFromSmiles("CC")  # Duplicate SMILES
        mol_b = Chem.MolFromSmiles("CO")
        # 4 attempts: 3 valid (1 duplicate), 1 None
        outputs: list[tuple[Mol, Mol] | None] = [
            (mol_a1, mol_a1),
            None,
            (mol_a2, mol_a2),
            (mol_b, mol_b),
        ]
        generator = self._make_dummy_generator(outputs)

        with tempfile.TemporaryDirectory() as tmpdir:
            generator.generate(
                data=_make_cloneable_data(),
                num_gen=4,
                rec_name="unit",
                with_print=False,
                root_path=tmpdir,
            )

            out_dir = Path(generator.out_dir)
            sdf_path = out_dir / "generated.sdf"
            smi_path = out_dir / "generated.smi"
            metrics_path = out_dir / "metrics.dir"

            # Check files exist
            self.assertTrue(sdf_path.exists(), "SDF file should exist")
            self.assertTrue(smi_path.exists(), "SMI file should exist")
            self.assertTrue(metrics_path.exists(), "Metrics file should exist")

            # Verify SDF contains 3 molecules
            sdf_text = sdf_path.read_text(encoding="utf-8")
            self.assertEqual(
                sdf_text.count("$$$$"),
                3,
                "SDF should contain 3 molecule blocks",
            )

            # Verify SMI contains 3 SMILES
            smiles = smi_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(smiles), 3, "SMI should contain 3 SMILES")

            # Verify metrics
            metrics = ast.literal_eval(metrics_path.read_text(encoding="utf-8"))
            self.assertAlmostEqual(
                metrics["validity"],
                0.75,
                places=6,
                msg="Validity should be 3/4 = 0.75",
            )
            self.assertAlmostEqual(
                metrics["unique"],
                2 / 3,
                places=6,
                msg="Unique should be 2/3 (2 unique out of 3 valid)",
            )
            self.assertIn("ring_size", metrics, "Metrics should include ring_size")

    def test_generate_handles_all_failures(self) -> None:
        """Test that generate() handles case when all generation attempts fail."""
        outputs: list[tuple[Mol, Mol] | None] = [None, None, None]
        generator = self._make_dummy_generator(outputs)

        with tempfile.TemporaryDirectory() as tmpdir:
            generator.generate(
                data=_make_cloneable_data(),
                num_gen=3,
                rec_name="unit_fail",
                with_print=False,
                root_path=tmpdir,
            )

            out_dir = Path(generator.out_dir)
            metrics_path = out_dir / "metrics.dir"

            self.assertTrue(metrics_path.exists(), "Metrics file should exist even on all failures")

            metrics = ast.literal_eval(metrics_path.read_text(encoding="utf-8"))
            self.assertAlmostEqual(
                metrics["validity"],
                0.0,
                places=6,
                msg="Validity should be 0 when all attempts fail",
            )
            self.assertAlmostEqual(
                metrics["unique"],
                0.0,
                places=6,
                msg="Unique should be 0 when no valid molecules",
            )

    def test_generate_all_unique(self) -> None:
        """Test that generate() correctly calculates uniqueness when all molecules are unique."""
        Chem = self.Chem

        mol_a = Chem.MolFromSmiles("CC")
        mol_b = Chem.MolFromSmiles("CO")
        mol_c = Chem.MolFromSmiles("CCO")
        outputs: list[tuple[Mol, Mol] | None] = [
            (mol_a, mol_a),
            (mol_b, mol_b),
            (mol_c, mol_c),
        ]
        generator = self._make_dummy_generator(outputs)

        with tempfile.TemporaryDirectory() as tmpdir:
            generator.generate(
                data=_make_cloneable_data(),
                num_gen=3,
                rec_name="unit_unique",
                with_print=False,
                root_path=tmpdir,
            )

            metrics_path = Path(generator.out_dir) / "metrics.dir"
            metrics = ast.literal_eval(metrics_path.read_text(encoding="utf-8"))

            self.assertAlmostEqual(
                metrics["validity"],
                1.0,
                places=6,
                msg="Validity should be 1.0 when all attempts succeed",
            )
            self.assertAlmostEqual(
                metrics["unique"],
                1.0,
                places=6,
                msg="Unique should be 1.0 when all molecules are unique",
            )


if __name__ == "__main__":
    unittest.main()
