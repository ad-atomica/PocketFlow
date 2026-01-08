from __future__ import annotations

from typing import Any, Protocol

from torch import Tensor

# Type alias for scalar/vector feature tuple: (scalar: [N, F_sca], vector: [N, F_vec, 3])
type ScalarVectorFeatures = tuple[Tensor, Tensor]

type BottleneckSpec = tuple[int, int]

# Type alias for edge index tensor: shape (2, E), dtype long
type EdgeIndex = Tensor


class TrainingBatch(Protocol):
    """Structural protocol for the training-time batch expected by `PocketFlow`.

    Instances are typically `torch_geometric.data.Data`/`Batch` objects produced
    by the transforms in `pocket_flow/utils/transform.py` and
    `ComplexDataTrajectory.from_steps`.

    Most tensors follow a small set of indexing conventions:
      - **Composed/complex tensors** (`cpx_*`) are indexed in the composed
        space of atoms. Ligand-context nodes are identified by
        `idx_ligand_ctx_in_cpx` (not necessarily `0..N_ctx-1`).
      - **Query tensors** (`y_pos`, `edge_query_index_*`, `pos_query_knn_*`) are
        indexed in the space of positions being predicted for the *next* atom(s)
        at this generation/training step.

    Attributes:
        cpx_feature:
            Per-node scalar features for the composed system, shape
            ``(N_cpx, F_in)``. Cast to float before embedding.
        cpx_pos:
            Cartesian coordinates for composed nodes, shape ``(N_cpx, 3)``.
        idx_ligand_ctx_in_cpx:
            Indices of ligand-context nodes within composed tensors,
            shape ``(N_ctx,)``.
        idx_protein_in_cpx:
            Indices of protein nodes within composed tensors, shape
            ``(N_prot,)``.
        cpx_edge_index:
            Edge indices for the composed context graph, shape ``(2, E_cpx)``,
            using PyG default ordering (source→target).
        cpx_edge_feature:
            Per-edge feature tensor for the composed graph, commonly one-hot bond
            types with shape ``(E_cpx, 4)`` (including a "no-bond" channel).
        ligand_frontier:
            Binary labels for ligand-context nodes indicating the ligand frontier
            / focal sites, shape ``(N_ctx,)`` (bool or 0/1).
        apo_protein_idx:
            Indices selecting apo protein nodes to score for "surface focal"
            supervision, shape ``(N_apo,)``. May be empty.
        candidate_focal_label_in_protein:
            Binary labels over `apo_protein_idx` nodes, shape ``(N_apo,)``.
            May be empty when `apo_protein_idx` is empty.
        atom_label:
            Integer class labels for the next atom type, shape ``(N_query,)``.
        focal_idx_in_context_selected:
            Indices of the chosen focal/attachment atom(s) in composed space,
            shape ``(N_query,)``.
        y_pos:
            Target next-atom coordinates, shape ``(N_query, 3)``.
        edge_label:
            Integer labels for queried edges (bond types), shape ``(E_query,)``.
            In this codepath, labels are one-hot encoded with ``num_classes=4``.
        edge_query_index_0, edge_query_index_1:
            Index vectors defining queried edges between query atoms and composed
            context atoms:
              - ``edge_query_index_0``: indices into `y_pos`, shape ``(E_query,)``
              - ``edge_query_index_1``: indices into `cpx_pos`, shape ``(E_query,)``
        pos_query_knn_edge_idx_0, pos_query_knn_edge_idx_1:
            kNN edges from composed nodes to query positions (source→target),
            used to build conditioning context for edge prediction:
              - ``pos_query_knn_edge_idx_0``: indices into `cpx_pos`, shape ``(E_knn,)``
              - ``pos_query_knn_edge_idx_1``: indices into `y_pos`, shape ``(E_knn,)``
        index_real_cps_edge_for_atten:
            Pair indices for edge-attention over the queried edges, stored as a
            stacked tensor of shape ``(2, E_att)`` and split into a tuple before
            passing to `BondFlow`.
        tri_edge_index: (2, E_tri)
            Pair indices (in composed node space) defining "triangle edges" used
            by edge-attention, stored in source→target order.
        tri_edge_feat:
            One-hot triangle-edge relation features, shape ``(E_att, 5)`` where
            the 5 channels correspond to relation types ``{-1, 0, 1, 2, 3}``
            (see `pocket_flow/utils/transform_utils.py:get_tri_edges`).
    """

    cpx_feature: Tensor
    cpx_pos: Tensor
    idx_ligand_ctx_in_cpx: Tensor
    idx_protein_in_cpx: Tensor
    cpx_edge_index: Tensor
    cpx_edge_feature: Tensor
    ligand_frontier: Tensor
    apo_protein_idx: Tensor
    candidate_focal_label_in_protein: Tensor
    atom_label: Tensor
    focal_idx_in_context_selected: Tensor
    y_pos: Tensor
    edge_label: Tensor
    edge_query_index_0: Tensor
    edge_query_index_1: Tensor
    pos_query_knn_edge_idx_0: Tensor
    pos_query_knn_edge_idx_1: Tensor
    index_real_cps_edge_for_atten: Tensor
    tri_edge_index: EdgeIndex
    tri_edge_feat: Tensor
    num_graphs: int

    def to(self, *args: Any, **kwargs: Any) -> TrainingBatch: ...


class PocketFlowEncoderConfig(Protocol):
    edge_channels: int
    num_interactions: int
    knn: int
    cutoff: float
    num_heads: int


class PocketFlowFocalNetConfig(Protocol):
    hidden_dim_sca: int
    hidden_dim_vec: int


class PocketFlowAtomFlowConfig(Protocol):
    hidden_dim_sca: int
    hidden_dim_vec: int
    num_flow_layers: int


class PocketFlowPositionPredictorConfig(Protocol):
    num_filters: tuple[int, int]
    n_component: int


class PocketFlowEdgeFlowConfig(Protocol):
    edge_channels: int
    num_filters: tuple[int, int]
    num_bond_types: int
    num_heads: int
    cutoff: float
    num_flow_layers: int


class PocketFlowConfig(Protocol):
    deq_coeff: float
    hidden_channels: int
    hidden_channels_vec: int
    bottleneck: BottleneckSpec
    use_conv1d: bool

    protein_atom_feature_dim: int
    ligand_atom_feature_dim: int
    num_atom_type: int
    num_bond_types: int
    msg_annealing: bool

    encoder: PocketFlowEncoderConfig
    focal_net: PocketFlowFocalNetConfig
    atom_flow: PocketFlowAtomFlowConfig
    pos_predictor: PocketFlowPositionPredictorConfig
    edge_flow: PocketFlowEdgeFlowConfig
