"""
Autoregressive ligand generation for PocketFlow.

This module implements the core *sampling loop* used by PocketFlow to grow a
ligand inside a protein pocket, one atom at a time. At a high level, generation
repeats the following steps until termination:

1. Embed and encode the current protein+ligand context graph.
2. Choose a *focal* context atom (protein surface atom for the first step, then
   a ligand-context atom for subsequent steps).
3. Sample a new ligand atom type from a flow model conditioned on the context.
4. Sample a new 3D position for that atom.
5. Predict bonds from the new atom to nearby context atoms, with chemistry and
   distance heuristics to keep the intermediate molecule plausible.
6. If successful, append the new atom/bonds to the ligand-context graph and
   continue; otherwise, resample up to a fixed budget.

Key data structures:
    - The generation loop operates on a PyG :class:`torch_geometric.data.Data`
      instance that must contain a "composed" protein+ligand context graph
      (fields like ``cpx_pos``, ``cpx_feature``, ``cpx_edge_index``) as well as
      the current ligand context (fields like ``ligand_context_pos`` and
      ``ligand_context_bond_index``). The exact schema is defined by the data
      pipeline/transforms in :mod:`pocket_flow.utils.transform`.
    - RDKit is used to maintain an incremental editable molecule
      (:class:`rdkit.Chem.RWMol`) for early valence/alert checks and to write
      out generated structures.

Outputs:
    :meth:`Generate.generate` writes results to ``root_path/rec_name/<timestamp>/``:
      - ``generated.sdf``: concatenated SDF blocks for valid molecules
      - ``generated.smi``: SMILES (one per line) for valid molecules
      - ``metrics.dir``: a stringified dictionary of validity/uniqueness/ring stats

Notes:
    - Generation uses :func:`torch.no_grad` and samples from Gaussian priors
      whose scales are controlled by ``temperature``.
    - Several operations mutate the provided :class:`~torch_geometric.data.Data`
      object (e.g., appending to ligand context). Callers should pass a clone if
      they need to preserve the original.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Literal, TypedDict

import torch
from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import BondType, Mol
from torch import Tensor
from torch.distributions import Normal

from pocket_flow.gdbp_model.net_utils import embed_compose
from pocket_flow.gdbp_model.pocket_flow import PocketFlow
from pocket_flow.gdbp_model.types import ScalarVectorFeatures
from pocket_flow.utils.data import ComplexData
from pocket_flow.utils.file_utils import ensure_parent_dir_exists
from pocket_flow.utils.generate_utils import (
    add_ligand_atom_to_data,
    check_alert_structures,
    check_valency,
    data2mol,
    modify,
)
from pocket_flow.utils.metrics import RingSizeStats, empty_ring_size_stats, substructure
from pocket_flow.utils.neighbor_search import knn, radius
from pocket_flow.utils.transform_utils import get_tri_edges

RDLogger.DisableLog("rdApp.*")

type FocalResult = tuple[Tensor, Tensor] | Literal[False]
type BondResult = tuple[Chem.RWMol, Tensor, Tensor] | Literal[False]
type GenerateResult = tuple[Mol, Mol] | None


class GenerationStats(TypedDict):
    """Summary statistics for a generation run.

    This TypedDict defines the structure of metrics written by
    :meth:`Generate.generate` to ``metrics.dir``.

    Attributes:
        validity: Fraction of attempted generations that produced a valid
            molecule.
        unique: Fraction of valid molecules that are unique by SMILES.
        ring_size: Ring-size distribution and ring motif counts as computed by
            :func:`pocket_flow.utils.metrics.substructure`.
    """

    validity: float
    unique: float
    ring_size: RingSizeStats


class Generate:
    """Autoregressive sampler that grows a ligand in a pocket.

    The sampler wraps a trained :class:`pocket_flow.gdbp_model.PocketFlow`
    instance and uses it to propose (1) a focal context atom, (2) a new ligand
    atom type, (3) a new position, and (4) new bonds at each generation step.

    The implementation is intentionally stateful:
      - Configuration is provided at construction time.
      - Per-run state (priors, counters, resampling flags) is initialized inside
        :meth:`run`.

    Parameters in brief:
        - ``temperature`` controls the scale of Gaussian priors for the atom and
          bond flows.
        - ``choose_max`` toggles whether to deterministically choose the maximum
          probability for certain decisions (focal selection and atom types)
          versus sampling from candidates.
        - ``max_atom_num`` bounds the number of autoregressive steps.
    """

    model: PocketFlow
    transform: Callable[[ComplexData], ComplexData]
    temperature: tuple[float, float]
    atom_type_map: Sequence[int]
    num_bond_type: int
    max_atom_num: int
    focus_threshold: float
    max_double_in_6ring: int
    min_dist_inter_mol: float
    bond_length_range: Sequence[float]
    choose_max: bool
    hidden_channels: int
    knn: int
    num_workers: int
    device: torch.device
    bond_type_map: dict[int, BondType]
    out_dir: str

    # Generation state (set during run)
    prior_node: Normal
    prior_edge: Normal
    resample_edge_failed: bool
    check_node: bool
    resample_node: int

    def __init__(
        self,
        model: PocketFlow,
        transform: Callable[[ComplexData], ComplexData],
        temperature: tuple[float, float] = (1.0, 1.0),
        atom_type_map: Sequence[int] = (6, 7, 8, 9, 15, 16, 17, 35, 53),
        num_bond_type: int = 4,
        max_atom_num: int = 35,
        focus_threshold: float = 0.5,
        max_double_in_6ring: int = 0,
        min_dist_inter_mol: float = 3.0,
        bond_length_range: Sequence[float] = (1.0, 2.0),
        choose_max: bool = True,
        device: str | torch.device = "cuda:0",
        *,
        num_workers: int,
    ) -> None:
        """Create a generator wrapper around a trained PocketFlow model.

        Args:
            model: Trained :class:`~pocket_flow.gdbp_model.PocketFlow` model.
            transform: Post-step transform applied after a new ligand atom is
                appended to the context graph. This is typically the same
                transform pipeline used during training (e.g., to rebuild
                composed edges/features for the next step).
            temperature: Two temperatures ``(node_temp, edge_temp)`` controlling
                the Gaussian prior scales used by ``atom_flow`` and ``edge_flow``.
                Higher values increase sampling diversity.
            atom_type_map: Mapping from model atom-type indices to atomic
                numbers (e.g., index 0 → 6 for carbon).
            num_bond_type: Number of bond-type classes predicted by the edge
                flow. The code treats ``0`` as "no bond" and maps ``1..3`` to
                RDKit single/double/triple.
            max_atom_num: Maximum number of ligand atoms to attempt to generate.
            focus_threshold: Probability threshold used when selecting candidate
                focal atoms (when not using argmax selection).
            max_double_in_6ring: Passed to :func:`pocket_flow.utils.generate_utils.modify`
                to limit the number of double bonds in 6-membered rings during
                final RDKit post-processing.
            min_dist_inter_mol: Minimum distance used when placing the very
                first ligand atom relative to the protein context (an
                "inter-molecular" cutoff).
            bond_length_range: Inclusive-ish distance window used when placing
                atoms after the first step; candidates outside this range are
                rejected and resampled.
            choose_max: If `True`, use argmax for certain selections (and a
                small amount of sampling for focal choice on step 0). If `False`,
                sample from candidates more often.
            device: Torch device string or device object (e.g., ``"cuda:0"``, ``"cpu"``,
             ``"mps"``, or ``torch.device("cuda:0")``).
            num_workers: Worker count for PyG neighbor search.
        """
        self.model = model
        self.transform = transform
        if len(temperature) != 2:
            raise ValueError(
                f"temperature must have exactly 2 elements (node_temp, edge_temp), got {len(temperature)}"
            )
        self.temperature = temperature
        self.atom_type_map = atom_type_map
        self.num_bond_type = num_bond_type
        self.max_atom_num = max_atom_num
        self.focus_threshold = focus_threshold
        self.max_double_in_6ring = max_double_in_6ring
        self.min_dist_inter_mol = min_dist_inter_mol
        if bond_length_range[1] <= bond_length_range[0]:
            raise ValueError(
                f"bond_length_range[1] ({bond_length_range[1]}) must be > "
                f"bond_length_range[0] ({bond_length_range[0]})"
            )
        self.bond_length_range = bond_length_range
        self.choose_max = choose_max
        self.hidden_channels = model.config.hidden_channels
        self.knn = model.config.encoder.knn
        self.num_workers = num_workers
        self.device = torch.device(device)
        self.bond_type_map = {
            1: Chem.rdchem.BondType.SINGLE,
            2: Chem.rdchem.BondType.DOUBLE,
            3: Chem.rdchem.BondType.TRIPLE,
        }

    @staticmethod
    def _choose_focal(
        focal_net: Callable[[ScalarVectorFeatures, Tensor], Tensor],
        h_ctx: ScalarVectorFeatures,
        ctx_idx: Tensor,
        focus_threshold: float,
        choose_max: bool,
        surf_mask: Tensor | None = None,
    ) -> FocalResult:
        """Select candidate focal indices from context embeddings.

        The focal network predicts a (logit) score per candidate context node.
        This helper converts logits to probabilities and returns either:
          - The single best focal index (if ``choose_max``), or
          - All indices above ``focus_threshold`` (optionally restricted to a
            surface mask).

        Args:
            focal_net: Model module that maps ``(h_ctx, ctx_idx)`` to per-index
                logits. This is typically ``model.focal_net``.
            h_ctx: Encoded node features for the composed context graph.
            ctx_idx: Indices into the composed node set to consider as candidates.
            focus_threshold: Minimum sigmoid probability for a node to be
                considered a focal candidate when not using argmax.
            choose_max: Whether to choose the single best scoring node.
            surf_mask: Optional boolean mask over candidates used on the first
                step to restrict focal choices to protein surface atoms.

        Returns:
            ``(focal_idx_candidate, focal_prob)`` where both tensors are 1D and
            aligned. Returns ``False`` if no candidate passes the selection
            criteria.
        """
        focal_pred = focal_net(h_ctx, ctx_idx)
        focal_prob = torch.sigmoid(focal_pred).view(-1)

        if choose_max:
            max_idx = focal_pred.argmax()
            focal_idx_candidate = ctx_idx[max_idx].view(-1)
            focal_prob = focal_prob[max_idx].view(-1)
        else:
            if isinstance(surf_mask, Tensor) and surf_mask.sum() > 0:
                candidate_ctx = ctx_idx[surf_mask]
                focal_prob_surf = focal_prob[surf_mask]
                surf_focal_mask = focal_prob_surf > focus_threshold
                focal_idx_candidate = candidate_ctx[surf_focal_mask]
                focal_prob = focal_prob_surf[surf_focal_mask]
                if surf_focal_mask.sum() == 0:
                    return False
            else:
                focal_mask = (focal_prob >= focus_threshold).view(-1)
                focal_idx_candidate = ctx_idx[focal_mask]
                focal_prob = focal_prob[focal_mask]
                if focal_mask.sum() == 0:
                    return False

        return focal_idx_candidate, focal_prob

    @staticmethod
    def _safe_multinomial(
        weights: Tensor,
        fallback: Literal["uniform", "argmax", "false"] = "uniform",
    ) -> Tensor | Literal[False]:
        """Sample 1 index from weights without hard-crashing.

        `torch.multinomial` requires finite, non-negative weights whose sum is > 0.
        Masking candidate weights (e.g., distance windows) can produce an all-zero
        vector even when candidates exist; model outputs may also contain NaNs/Infs.
        """
        weights = weights.view(-1)
        if weights.numel() == 0:
            return False
        if weights.numel() == 1:
            return torch.zeros(1, dtype=torch.long, device=weights.device)

        weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        weights = torch.clamp(weights, min=0.0)
        total = weights.sum()
        if (not torch.isfinite(total)) or total.item() <= 0.0:
            if fallback == "false":
                return False
            if fallback == "argmax":
                return weights.argmax().view(1)
            if fallback == "uniform":
                return torch.randint(0, weights.numel(), (1,), device=weights.device)

        return torch.multinomial(weights, 1)

    @staticmethod
    def _assert_ctx_cpx_mapping(data: ComplexData) -> None:
        """Validate the ligand context to complex index mapping.

        Ensures that ``idx_ligand_ctx_in_cpx`` correctly maps ligand context atom
        indices into the composed complex graph index space. Specifically checks:
        - The number of mapping indices matches the number of ligand context atoms
        - All indices are non-negative
        - All indices are within bounds (less than the number of complex atoms)
        - All indices are unique (no duplicates)

        Args:
            data: ComplexData instance containing the mapping to validate.

        Raises:
            AssertionError: If any validation check fails.
        """
        idx = data.idx_ligand_ctx_in_cpx
        n_ctx = int(data.ligand_context_pos.size(0))
        assert idx.numel() == n_ctx
        if idx.numel() == 0:
            return
        n_cpx = int(data.cpx_pos.size(0))
        assert int(idx.min().item()) >= 0
        assert int(idx.max().item()) < n_cpx
        assert idx.unique().numel() == idx.numel()

    def choose_focal(
        self,
        h_cpx: ScalarVectorFeatures,
        cpx_index: Tensor,
        idx_ligand_ctx_in_cpx: Tensor,
        data: ComplexData,
        atom_idx: int,
    ) -> FocalResult:
        """Choose focal atom(s) for the current generation step.

        Step 0 (no ligand context yet):
            Candidates come from ``cpx_index`` (protein indices) and may be
            restricted to ``data.protein_surface_mask`` when present.

        Step > 0:
            Candidates come from ``idx_ligand_ctx_in_cpx`` and the method
            additionally enforces a simple *remaining valence* constraint by
            comparing ``data.ligand_context_valence`` against ``data.max_atom_valence``.

        Args:
            h_cpx: Encoded composed node features (protein + ligand context).
            cpx_index: Protein indices in the composed graph.
            idx_ligand_ctx_in_cpx: Ligand-context indices in the composed graph.
            data: Current data object (used for surface/valence checks).
            atom_idx: Current generation step (0-based).

        Returns:
            ``(focal_idx, focal_prob)`` describing the chosen candidates, or
            ``False`` if no valid focal exists.
        """
        if atom_idx == 0:
            surf_mask = data["protein_surface_mask"] if "protein_surface_mask" in data else None
            focal_out = self._choose_focal(
                self.model.focal_net,
                h_cpx,
                cpx_index,
                self.focus_threshold,
                self.choose_max,
                surf_mask=surf_mask,
            )
        else:
            focal_out = self._choose_focal(
                self.model.focal_net,
                h_cpx,
                idx_ligand_ctx_in_cpx,
                self.focus_threshold,
                True,
            )

        if focal_out is False:
            return False

        focal_idx_, focal_prob = focal_out

        # valence_check for focal atom
        focal_valence_check = False
        if data.ligand_context_element.size(0) > 3 and focal_idx_ is not False and atom_idx != 0:
            self._assert_ctx_cpx_mapping(data)
            n_cpx = int(data.cpx_pos.size(0))
            n_ctx = int(data.ligand_context_pos.size(0))
            cpx_to_ctx = torch.full((n_cpx,), -1, dtype=torch.long, device=focal_idx_.device)
            cpx_to_ctx[data.idx_ligand_ctx_in_cpx] = torch.arange(n_ctx, device=focal_idx_.device)
            focal_idx_ctx = cpx_to_ctx[focal_idx_]
            if (focal_idx_ctx < 0).any().item():
                return False
            max_valence = data.max_atom_valence[focal_idx_ctx]
            valence_in_ligand_context_focal = data.ligand_context_valence[focal_idx_ctx]
            valence_mask = max_valence > valence_in_ligand_context_focal
            focal_idx_ = focal_idx_[valence_mask]
            focal_prob = focal_prob[valence_mask]
            focal_valence_check = valence_mask.sum() == 0
            if focal_valence_check:
                return False

        if focal_valence_check:
            return False

        return focal_idx_, focal_prob

    def atom_generate(
        self,
        h_cpx: ScalarVectorFeatures,
        focal_idx: Tensor,
        focal_prob: Tensor,
        atom_idx: int,
    ) -> tuple[Tensor, Tensor]:
        """Sample a new atom type conditioned on the current context.

        The node flow is inverted from a Gaussian prior to produce logits over
        ``len(self.atom_type_map)`` atom-type classes.

        Selection behavior depends on ``self.choose_max`` and ``atom_idx``:
            - When ``choose_max`` and ``atom_idx == 0``: pick a likely atom type
              but sample the focal index from ``focal_prob`` (encourages diverse
              placements on the protein surface).
            - Otherwise: choose the argmax atom type (and keep focal indices as
              given).

        Args:
            h_cpx: Encoded composed node features.
            focal_idx: Candidate focal indices.
            focal_prob: Candidate focal probabilities (aligned with ``focal_idx``).
            atom_idx: Current generation step.

        Returns:
            ``(new_atom_type, focal_idx_)`` where ``new_atom_type`` is a scalar
            tensor containing the chosen atom-type *index* into
            ``self.atom_type_map``, and ``focal_idx_`` is the focal index used
            for position/bond generation.
        """
        z_atom = self.prior_node.sample([focal_idx.size(0)])
        x_atom = self.model.atom_flow.reverse(z_atom, h_cpx, focal_idx)

        if self.choose_max:
            if atom_idx == 0:
                new_atom_type_prob, new_atom_type_pool = torch.max(x_atom, -1)
                new_atom_idx_with_max_prob = torch.flip(new_atom_type_prob.argsort(), dims=[0])
                new_atom_type_pool = new_atom_type_pool[new_atom_idx_with_max_prob]

                focal_idx_sorted = focal_idx[new_atom_idx_with_max_prob]
                focal_prob_sorted = focal_prob[new_atom_idx_with_max_prob]
                focal_choose_idx = self._safe_multinomial(focal_prob_sorted, fallback="uniform")
                focal_idx_ = focal_idx_sorted[focal_choose_idx]
                new_atom_type = new_atom_type_pool[focal_choose_idx]
            else:
                _, new_atom_type = torch.max(x_atom, -1)
                focal_idx_ = focal_idx
        else:
            new_atom_type_prob, new_atom_type_pool = torch.max(x_atom, -1)
            new_atom_idx_with_max_prob = torch.flip(new_atom_type_prob.argsort(), dims=[0])
            new_atom_type_pool = new_atom_type_pool[new_atom_idx_with_max_prob]

            focal_idx_sorted = focal_idx[new_atom_idx_with_max_prob]
            focal_prob_sorted = focal_prob[new_atom_idx_with_max_prob]
            focal_choose_idx = self._safe_multinomial(focal_prob_sorted, fallback="uniform")
            focal_idx_ = focal_idx_sorted[focal_choose_idx]
            new_atom_type = new_atom_type_pool[focal_choose_idx]

        return new_atom_type, focal_idx_

    def pos_generate(
        self,
        h_cpx: ScalarVectorFeatures,
        atom_type_emb: Tensor,
        focal_idx: Tensor,
        cpx_pos: Tensor,
        atom_idx: int,
    ) -> Tensor | Literal[False]:
        """Sample a 3D position for the next ligand atom.

        The position predictor proposes a pool of candidate positions relative
        to the focal atom(s) and returns mixture weights ``pi`` for sampling.

        Distance heuristics:
            - For ``atom_idx == 0`` (first atom), candidates closer than
              ``min_dist_inter_mol`` are filtered out. If none remain, the
              farthest candidate is chosen as a fallback.
            - For later steps, candidates must lie within
              ``bond_length_range``; if none remain, the step fails.

        Args:
            h_cpx: Encoded composed node features.
            atom_type_emb: Embedding of the chosen atom type, shape
                ``(1, hidden_channels)``.
            focal_idx: Focal index tensor, shape ``(1,)`` in typical usage.
            cpx_pos: Composed node coordinates.
            atom_idx: Current generation step.

        Returns:
            A position tensor of shape ``(1, 3)`` (as produced by the model), or
            ``False`` if no candidate satisfies the distance constraints.
        """
        new_relative_pos, new_abs_pos, _sigma, pi = self.model.pos_predictor(
            h_cpx, focal_idx, cpx_pos, atom_type_emb=atom_type_emb
        )
        new_relative_pos = new_relative_pos.view(-1, 3)
        new_abs_pos = new_abs_pos.view(-1, 3)
        pi = pi.view(-1)
        dist = torch.norm(new_relative_pos, p=2, dim=-1)

        if atom_idx != 0:
            dist_mask = (dist > self.bond_length_range[0]) & (dist < self.bond_length_range[1])
            new_pos_to_add_ = new_abs_pos[dist_mask]
            check_pos = new_pos_to_add_.size(0) != 0
            if check_pos:
                pi_ = pi[dist_mask]
                pos_choose_idx = self._safe_multinomial(pi_, fallback="false")
                if pos_choose_idx is False:
                    return False
                new_pos_to_add = new_pos_to_add_[pos_choose_idx]
                return new_pos_to_add
            else:
                return False
        else:
            dist_mask = dist > self.min_dist_inter_mol  # inter-molecular dist cutoff
            new_pos_to_add_ = new_abs_pos[dist_mask]
            check_pos = new_pos_to_add_.size(0) != 0
            if check_pos:
                pi_ = pi[dist_mask]
                pos_choose_idx = self._safe_multinomial(pi_, fallback="false")
                if pos_choose_idx is False:
                    return False
                new_pos_to_add = new_pos_to_add_[pos_choose_idx]
                return new_pos_to_add
            else:
                new_pos_to_add = new_abs_pos[torch.argmax(dist).view(-1)]
                return new_pos_to_add

    def bond_generate(
        self,
        h_cpx: ScalarVectorFeatures,
        data: ComplexData,
        new_pos_to_add: Tensor,
        atom_type_emb: Tensor,
        atom_idx: int,
        rw_mol: Chem.RWMol,
    ) -> BondResult:
        """Predict bonds from the new atom to nearby context atoms.

        For non-initial steps, this method:
          - Builds candidate edges from ``new_pos_to_add`` to ligand context
            atoms within radius 4 Å.
          - Computes auxiliary kNN and triangle features via
            :func:`pocket_flow.utils.transform_utils.get_tri_edges`.
          - Samples an edge latent from ``self.prior_edge`` and inverts the edge
            flow to get bond-type logits.
          - Adds predicted bonds to ``rw_mol`` and validates via
            :func:`pocket_flow.utils.generate_utils.check_valency`.

        If the predicted bonds violate simple distance or valence constraints,
        the method removes the added bonds and resamples up to 50 times. On
        repeated failure, it also removes the newly added atom from ``rw_mol``
        and returns ``False``.

        Args:
            h_cpx: Encoded composed node features.
            data: Current context graph (used for neighbor search and features).
            new_pos_to_add: Position of the new atom, shape ``(1, 3)``.
            atom_type_emb: Embedding of the chosen atom type.
            atom_idx: Current generation step.
            rw_mol: Editable RDKit molecule tracking the current partial ligand.

        Returns:
            ``(rw_mol, new_edge_idx, new_bond_type_to_add)`` where
            ``new_edge_idx`` is a ``(2, E)`` tensor indexing into the ligand
            context (ctx space), and ``new_bond_type_to_add`` contains integer
            bond types.
            Returns ``False`` if bond prediction repeatedly fails.
        """
        if atom_idx == 0:
            new_edge_idx = torch.empty([2, 0], dtype=torch.long)
            new_bond_type_to_add = torch.empty([0], dtype=torch.long)
        else:
            self._assert_ctx_cpx_mapping(data)
            edge_index_query_ctx = radius(
                data.ligand_context_pos, new_pos_to_add, r=4.0, num_workers=self.num_workers
            )
            if edge_index_query_ctx.size(1) == 0:
                self.resample_edge_failed = True
                rw_mol.RemoveAtom(atom_idx)
                return False
            edge_index_query_cpx = torch.stack(
                [
                    edge_index_query_ctx[0],
                    data.idx_ligand_ctx_in_cpx[edge_index_query_ctx[1]],
                ],
                dim=0,
            )
            pos_query_knn_edge_idx = knn(
                x=data.cpx_pos,
                y=new_pos_to_add,
                k=self.model.config.encoder.knn,
                num_workers=self.num_workers,
            )
            pos_query_knn_edge_idx = pos_query_knn_edge_idx[[1, 0]]
            index_real_cps_edge_for_atten, tri_edge_index, tri_edge_feat = get_tri_edges(
                edge_index_query_ctx,
                new_pos_to_add,
                data.idx_ligand_ctx_in_cpx,
                data.ligand_context_bond_index,
                data.ligand_context_bond_type,
            )
            resample_edge = 0
            no_bond = True

            while no_bond:
                if resample_edge >= 50:
                    self.resample_edge_failed = True
                    rw_mol.RemoveAtom(atom_idx)
                    return False

                latent_edge = self.prior_edge.sample([edge_index_query_ctx.size(1)])
                edge_latent = self.model.edge_flow.reverse(
                    edge_latent=latent_edge,
                    pos_query=new_pos_to_add,
                    edge_index_query=edge_index_query_cpx,
                    cpx_pos=data.cpx_pos,
                    node_attr_compose=h_cpx,
                    edge_index_q_cps_knn=pos_query_knn_edge_idx,
                    index_real_cps_edge_for_atten=(
                        index_real_cps_edge_for_atten[0],
                        index_real_cps_edge_for_atten[1],
                    ),
                    tri_edge_index=tri_edge_index,
                    tri_edge_feat=tri_edge_feat,
                    atom_type_emb=atom_type_emb,
                    annealing=False,
                )
                edge_pred_type = edge_latent.argmax(-1)
                edge_pred_mask = edge_pred_type > 0

                if edge_pred_mask.sum() > 0:
                    new_bond_type_to_add = edge_pred_type[edge_pred_mask]
                    new_edge_idx = edge_index_query_ctx[:, edge_pred_mask]
                    new_edge_idx_cpx = edge_index_query_cpx[:, edge_pred_mask]

                    new_edge_vec = new_pos_to_add[new_edge_idx_cpx[0]] - data.cpx_pos[new_edge_idx_cpx[1]]
                    new_edge_dist = torch.norm(new_edge_vec, p=2, dim=-1)

                    if (new_edge_dist > self.bond_length_range[1]).sum() > 0:
                        if resample_edge >= 50:
                            self.resample_edge_failed = True
                            rw_mol.RemoveAtom(atom_idx)
                            return False
                        else:
                            resample_edge += 1
                            continue

                    for ix in range(new_edge_idx.size(1)):
                        i, j = new_edge_idx[:, ix].tolist()
                        bond_type_int = int(new_bond_type_to_add[ix].item())
                        rw_mol.AddBond(atom_idx, j, self.bond_type_map[bond_type_int])

                    valency_valid = check_valency(rw_mol)
                    if valency_valid:
                        no_bond = False
                        break
                    else:
                        for ix in range(new_edge_idx.size(1)):
                            i, j = new_edge_idx[:, ix].tolist()
                            rw_mol.RemoveBond(atom_idx, j)
                        resample_edge += 1
                        if resample_edge >= 50:
                            self.resample_edge_failed = True
                            rw_mol.RemoveAtom(atom_idx)
                            return False
                else:
                    resample_edge += 1

        return rw_mol, new_edge_idx, new_bond_type_to_add

    def run(self, data: ComplexData) -> GenerateResult:
        """Run a single autoregressive generation attempt.

        This method performs up to ``self.max_atom_num`` growth steps and stops
        early when:
          - No valid focal can be found,
          - Position sampling fails too many times,
          - Bond prediction fails too many times.

        On success, the final ligand context graph is converted back to an RDKit
        molecule and then post-processed with :func:`pocket_flow.utils.generate_utils.modify`.

        Args:
            data: Input composed graph and ligand context. This object is moved
                across devices and mutated during the run.

        Returns:
            ``(modified_mol, raw_mol)`` on success, where:
              - ``raw_mol`` is the direct RDKit reconstruction from the final
                ligand context graph.
              - ``modified_mol`` is the post-processed version returned by
                :func:`~pocket_flow.utils.generate_utils.modify`.
            Returns ``None`` when molecule reconstruction/post-processing fails
            or the run terminates early.
        """
        data.max_atom_valence = torch.empty(0, dtype=torch.long)
        data = data.to(self.device)

        with torch.no_grad():
            self.prior_node = Normal(
                torch.zeros(len(self.atom_type_map), device=data.cpx_pos.device),
                self.temperature[0] * torch.ones(len(self.atom_type_map), device=data.cpx_pos.device),
            )
            self.prior_edge = Normal(
                torch.zeros(self.num_bond_type, device=data.cpx_pos.device),
                self.temperature[1] * torch.ones(self.num_bond_type, device=data.cpx_pos.device),
            )

            rw_mol = Chem.RWMol()

            for atom_idx in range(self.max_atom_num):
                data = data.to(self.device)
                h_cpx_list = embed_compose(
                    data.cpx_feature.float(),
                    data.cpx_pos,
                    data.idx_ligand_ctx_in_cpx,
                    data.idx_protein_in_cpx,
                    self.model.ligand_atom_emb,
                    self.model.protein_atom_emb,
                    self.model.emb_dim,
                )
                h_cpx = (h_cpx_list[0], h_cpx_list[1])

                # encoding context
                h_cpx = self.model.encoder(
                    node_attr=h_cpx,
                    pos=data.cpx_pos,
                    edge_index=data.cpx_edge_index,
                    edge_feature=data.cpx_edge_feature,
                )

                self.resample_edge_failed = False
                self.check_node = True
                self.resample_node = 0

                while self.check_node:
                    if self.resample_node > 50:
                        break

                    # choose focal
                    focal_out = self.choose_focal(
                        h_cpx,
                        data.idx_protein_in_cpx,
                        data.idx_ligand_ctx_in_cpx,
                        data,
                        atom_idx,
                    )
                    if focal_out is False:
                        break
                    else:
                        focal_idx, focal_prob = focal_out

                    # generate atom
                    new_atom_type, focal_idx = self.atom_generate(h_cpx, focal_idx, focal_prob, atom_idx)

                    # get position of new atom
                    atom_type_emb = self.model.atom_type_embedding(new_atom_type).view(
                        -1, self.hidden_channels
                    )
                    new_pos_to_add = self.pos_generate(
                        h_cpx, atom_type_emb, focal_idx, data.cpx_pos, atom_idx
                    )
                    if new_pos_to_add is False:
                        self.resample_node += 1
                        continue
                    else:
                        atom_type_idx = int(new_atom_type.item())
                        rw_mol.AddAtom(Chem.Atom(self.atom_type_map[atom_type_idx]))

                    # generate bonds
                    bond_out = self.bond_generate(
                        h_cpx, data, new_pos_to_add, atom_type_emb, atom_idx, rw_mol
                    )
                    if bond_out is not False:
                        rw_mol, new_edge_idx, new_bond_type_to_add = bond_out
                        has_alert = check_alert_structures(
                            rw_mol,
                            [
                                "[O]-[O]",
                                "[N]-[O,Br,Cl,I,F,P]",
                                "[S,P]-[Br,Cl,I,F]",
                                "[P]-[O]-[P]",
                                "[Br,Cl,I,F]-[Br,Cl,I,F]",
                            ],
                        )
                        if has_alert:
                            for ix in range(new_edge_idx.size(1)):
                                i, j = new_edge_idx[:, ix].tolist()
                                rw_mol.RemoveBond(atom_idx, j)
                            rw_mol.RemoveAtom(atom_idx)
                            self.resample_node += 1
                            continue
                        else:
                            break
                    if self.resample_edge_failed:
                        break

                if self.resample_edge_failed or self.resample_node > 50 or focal_out is False:
                    break
                else:
                    data = data.to("cpu")
                    data = add_ligand_atom_to_data(
                        data,
                        new_pos_to_add.to("cpu"),
                        new_atom_type.to("cpu"),
                        new_edge_idx.to("cpu"),
                        new_bond_type_to_add.to("cpu"),
                        type_map=tuple(self.atom_type_map),
                    )
                    data = self.transform(data)

        try:
            mol = data2mol(data)
            modified_mol = modify(mol, max_double_in_6ring=self.max_double_in_6ring)
            return modified_mol, mol
        except Exception:
            mol_ = rw_mol.GetMol()
            print("Invalid mol: ", Chem.MolToSmiles(mol_))
            try:
                _ = data2mol(data, raise_error=False)
            except Exception:
                pass
            return None

    def generate(
        self,
        data: ComplexData,
        num_gen: int = 100,
        rec_name: str = "recptor",
        with_print: bool = True,
        root_path: str = "gen_results",
    ) -> None:
        """Generate multiple molecules and write them (plus metrics) to disk.

        For each attempt, :meth:`run` is called on a detached clone of ``data``.
        Molecules that successfully round-trip to RDKit are appended to output
        files in a timestamped directory:

        - ``generated.sdf``: SDF blocks for each valid molecule.
        - ``generated.smi``: SMILES for each valid molecule.
        - ``metrics.dir``: stringified dict with validity/uniqueness and ring stats.

        Args:
            data: Input composed graph and ligand context for generation. The
                method clones and detaches this object per attempt.
            num_gen: Number of generation attempts.
            rec_name: Label used to namespace outputs under ``root_path``.
            with_print: If `True`, print SMILES for each valid molecule.
            root_path: Root directory for generation outputs.
        """
        date = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        out_dir = root_path + "/" + rec_name + "/" + date + "/"
        self.out_dir = out_dir
        ensure_parent_dir_exists(out_dir)

        valid_mol = []
        smiles_list = []
        valid_counter = 0

        for _i in range(num_gen):
            data_clone = data.clone().detach()
            out = self.run(data_clone)
            mol = None
            if out:
                mol, _ = out
            del data_clone

            if mol is not None:
                mol.SetProp("_Name", f"No_{valid_counter}-{out_dir}")
                smi = Chem.MolToSmiles(mol)
                if with_print:
                    print(smi)
                with open(out_dir + "generated.sdf", "a") as sdf_writer:
                    mol_block = Chem.MolToMolBlock(mol)
                    sdf_writer.write(mol_block + "\n$$$$\n")
                with open(out_dir + "generated.smi", "a") as smi_writer:
                    smi_writer.write(smi + "\n")
                smiles_list.append(smi)
                valid_mol.append(mol)
                valid_counter += 1

        print(len(smiles_list))
        print(len(set(smiles_list)))
        print(f"Validity: {len(smiles_list) / num_gen:.4f}")

        if len(smiles_list) == 0:
            unique_value = 0.0
            print(f"Unique: {unique_value:.4f}")
        else:
            unique_value = len(set(smiles_list)) / len(smiles_list)
            print(f"Unique: {unique_value:.4f}")

        out_statistic: GenerationStats = {
            "validity": len(smiles_list) / num_gen,
            "unique": unique_value,
            "ring_size": empty_ring_size_stats() if len(valid_mol) == 0 else substructure([valid_mol]),
        }

        with open(out_dir + "metrics.dir", "w") as fw:
            fw.write(str(out_statistic))
