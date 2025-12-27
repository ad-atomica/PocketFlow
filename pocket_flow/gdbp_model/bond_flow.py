"""Bond-type normalising flow.

This module implements a conditional, elementwise affine normalising flow over
edge/bond-type features. The flow maps a dequantised categorical edge
representation x ∈ R^(B+1) (one-hot over bond types plus 'no-bond', with uniform
dequantisation noise) to a base latent variable z ∈ R^(B+1) (typically standard
Gaussian), conditioned on geometry- and context-dependent edge features produced
by GDBP-based networks.

The flow is diagonal in the edge feature dimension, so the log-determinant of
the Jacobian is tractable: the per-edge scalar log|det J| is the sum of the
elementwise log-scale terms over the last dimension.
"""

from __future__ import annotations

from typing import override

import torch
from torch import Tensor
from torch.nn import Module, ModuleList, Sequential

from pocket_flow.gdbp_model.layers import (
    AttentionEdges,
    GDBLinear,
    GDBPerceptronVN,
    MessageAttention,
    MessageModule,
    ST_GDBP_Exp,
)
from pocket_flow.gdbp_model.net_utils import EdgeExpansion, GaussianSmearing

# Type alias for scalar/vector feature tuple: (scalar: [N, F_sca], vector: [N, F_vec, 3])
type ScalarVectorFeatures = tuple[Tensor, Tensor]


def _has_edges(edge_index: Tensor) -> bool:
    """Return True iff `edge_index` is a [2, E] integer tensor with E > 0."""
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        return False
    return edge_index.size(1) > 0


class PositionEncoder(Module):
    """Encode query positions into scalar/vector features using complex context.

    Given query atom positions and query→complex neighbour edges, this encoder
    aggregates messages from complex atoms into per-query features using
    message passing and attention.

    Inputs/outputs follow the convention used throughout the model:
      - scalar features: (N, F_sca)
      - vector features: (N, F_vec, 3)

    The returned features are used as conditioning context for downstream edge
    (bond) prediction and the bond flow. This module itself is not a flow; it is
    a deterministic conditioner.
    """

    message_module: MessageModule
    message_att: MessageAttention
    distance_expansion: GaussianSmearing
    vector_expansion: EdgeExpansion
    root_lin: GDBLinear
    root_vector_expansion: EdgeExpansion

    def __init__(
        self,
        in_sca: int,
        in_vec: int,
        edge_channels: int,
        num_filters: tuple[int, int],
        bottleneck: int = 1,
        cutoff: float = 10.0,
        num_heads: int = 1,
        use_conv1d: bool = False,
    ) -> None:
        """Initialize the PositionEncoder.

        Args:
            in_sca: Number of input scalar channels.
            in_vec: Number of input vector channels.
            edge_channels: Number of edge feature channels.
            num_filters: Tuple of (scalar_filters, vector_filters) for output.
            bottleneck: Bottleneck factor for linear layers.
            cutoff: Distance cutoff for neighbor interactions.
            num_heads: Number of attention heads.
            use_conv1d: Whether to use 1D convolutions instead of linear layers.
        """
        super().__init__()
        self.message_module = MessageModule(
            in_sca,
            in_vec,
            edge_channels,
            edge_channels,
            num_filters[0],
            num_filters[1],
            bottleneck,
            cutoff,
            use_conv1d=use_conv1d,
        )
        self.message_att = MessageAttention(
            num_filters[0],
            num_filters[1],
            num_filters[0],
            num_filters[1],
            bottleneck=bottleneck,
            num_heads=num_heads,
            use_conv1d=use_conv1d,
        )
        self.distance_expansion = GaussianSmearing(stop=cutoff, num_gaussians=edge_channels)
        self.vector_expansion = EdgeExpansion(edge_channels)
        self.root_lin = GDBLinear(
            in_sca, in_vec, num_filters[0], num_filters[1], bottleneck=bottleneck, use_conv1d=use_conv1d
        )
        self.root_vector_expansion = EdgeExpansion(in_vec)

    @override
    def forward(
        self,
        pos_query: Tensor,
        edge_index_q_cps_knn: Tensor,
        cpx_pos: Tensor,
        node_attr_compose: ScalarVectorFeatures,
        atom_type_emb: Tensor,
        annealing: bool = False,
    ) -> ScalarVectorFeatures:
        """Compute query features conditioned on nearby complex atoms.

        Args:
            pos_query:
                Query atom positions, shape (N_query, 3).
            edge_index_q_cps_knn:
                Query→complex neighbour indices, shape (2, E_knn), where
                edge_index_q_cps_knn[0] are query indices and
                edge_index_q_cps_knn[1] are complex indices.
            cpx_pos:
                Complex atom positions, shape (N_complex, 3).
            node_attr_compose:
                Complex node features as (scalar, vector):
                  - scalar: (N_complex, F_sca)
                  - vector: (N_complex, F_vec, 3)
            atom_type_emb:
                Query atom-type embeddings (used as scalar root features),
                shape (N_query, F_embed).
            annealing:
                If True, apply distance-based message annealing.

        Returns:
            (scalar, vector) query features:
              - scalar: (N_query, num_filters[0])
              - vector: (N_query, num_filters[1], 3)
        """
        vec_ij = pos_query[edge_index_q_cps_knn[0]] - cpx_pos[edge_index_q_cps_knn[1]]
        dist_ij = torch.norm(vec_ij, p=2, dim=-1).view(-1, 1)  # [E, 1]
        edge_ij: ScalarVectorFeatures = (self.distance_expansion(dist_ij), self.vector_expansion(vec_ij))

        root_vec_ij = self.root_vector_expansion(pos_query)
        y_root_sca, y_root_vec = self.root_lin([atom_type_emb, root_vec_ij])
        x: ScalarVectorFeatures = (y_root_sca, y_root_vec)

        h_q = self.message_module(
            node_attr_compose, edge_ij, edge_index_q_cps_knn[1], dist_ij, annealing=annealing
        )
        y = self.message_att(x, h_q, edge_index_q_cps_knn[0])
        return y


class BondFlow(Module):
    """Conditional normalising flow for bond-type (edge) features.

    This model defines an elementwise affine bijection between a dequantised
    categorical edge representation x ∈ R^(B+1) and a base latent variable
    z ∈ R^(B+1), conditioned on learned edge context features.

    Conditioning pipeline:
      1) `PositionEncoder` computes per-query node features from the complex.
      2) For each queried edge (query i, complex j), build edge features from
         node features and relative geometry, and refine them via attention.
      3) Each flow layer predicts elementwise (log s, t) from these edge
         conditioning features and applies a diagonal affine transform.

    Flow transform (data→latent, used for training density):
        z = (x + t(h)) ⊙ exp(log s(h))

    Inverse transform (latent→data, used for sampling):
        x = z ⊙ exp(-log s(h)) - t(h)

    The log-Jacobian is diagonal; this implementation returns elementwise
    log|∂z/∂x| contributions of shape (E, B+1). Sum over the last dimension to
    obtain a scalar log|det J| per edge.
    """

    num_bond_types: int
    num_st_layers: int
    pos_encoder: PositionEncoder
    distance_expansion_3A: GaussianSmearing
    vector_expansion: EdgeExpansion
    nn_edge_ij: Sequential
    edge_feat: Sequential
    edge_atten: AttentionEdges
    flow_layers: ModuleList

    def __init__(
        self,
        in_sca: int,
        in_vec: int,
        edge_channels: int,
        num_filters: tuple[int, int],
        num_bond_types: int = 3,
        num_heads: int = 4,
        cutoff: float = 10.0,
        num_st_layers: int = 6,
        bottleneck: int = 1,
        use_conv1d: bool = False,
    ) -> None:
        """Initialize the BondFlow model.

        Args:
            in_sca: Number of input scalar channels.
            in_vec: Number of input vector channels.
            edge_channels: Number of edge feature channels.
            num_filters: Tuple of (scalar_filters, vector_filters).
            num_bond_types: Number of bond types (default 3: single, double, triple).
            num_heads: Number of attention heads.
            cutoff: Distance cutoff for neighbor interactions.
            num_st_layers: Number of scale-translation flow layers.
            bottleneck: Bottleneck factor for linear layers.
            use_conv1d: Whether to use 1D convolutions instead of linear layers.
        """
        super().__init__()
        self.num_bond_types = num_bond_types
        self.num_st_layers = num_st_layers

        # Query encoder
        self.pos_encoder = PositionEncoder(
            in_sca,
            in_vec,
            edge_channels,
            num_filters,
            cutoff=cutoff,
            # num_heads=num_heads,  # TODO: this was not used in the original code, but should it be ?
            bottleneck=bottleneck,
            use_conv1d=use_conv1d,
        )

        # Edge prediction networks
        self.distance_expansion_3A = GaussianSmearing(stop=3.0, num_gaussians=edge_channels)
        self.vector_expansion = EdgeExpansion(edge_channels)
        self.nn_edge_ij = Sequential(
            GDBPerceptronVN(edge_channels, edge_channels, num_filters[0], num_filters[1]),
            GDBLinear(num_filters[0], num_filters[1], num_filters[0], num_filters[1]),
        )
        self.edge_feat = Sequential(
            GDBPerceptronVN(
                num_filters[0] * 2 + in_sca, num_filters[1] * 2 + in_vec, num_filters[0], num_filters[1]
            ),
            GDBLinear(num_filters[0], num_filters[1], num_filters[0], num_filters[1]),
        )
        self.edge_atten = AttentionEdges(num_filters, num_filters, num_heads, num_bond_types)

        # Normalizing flow layers
        self.flow_layers = ModuleList()
        for _ in range(num_st_layers):
            flow_layer = ST_GDBP_Exp(
                num_filters[0],
                num_filters[1],
                num_bond_types + 1,  # +1 for no-bond type
                num_filters[1],
                bottleneck=bottleneck,
                use_conv1d=use_conv1d,
            )
            self.flow_layers.append(flow_layer)

    @override
    def forward(
        self,
        z_edge: Tensor,
        pos_query: Tensor,
        edge_index_query: Tensor,
        cpx_pos: Tensor,
        node_attr_compose: ScalarVectorFeatures,
        edge_index_q_cps_knn: Tensor,
        atom_type_emb: Tensor,
        index_real_cps_edge_for_atten: tuple[Tensor, Tensor],
        tri_edge_index: tuple[Tensor, Tensor],
        tri_edge_feat: Tensor,
        annealing: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """Map dequantised edge/bond-type features to latent space.

        Args:
            z_edge:
                Dequantised categorical edge representation in data space,
                shape (E, B+1), where B = num_bond_types and the extra channel
                represents 'no-bond'.
                (Despite the name, this tensor is treated as x in the
                change-of-variables direction used for training.)
            pos_query:
                Query atom positions, shape (N_query, 3).
            edge_index_query:
                Queried edges to score, shape (2, E), where row 0 are query
                indices and row 1 are complex indices.
            cpx_pos:
                Complex atom positions, shape (N_complex, 3).
            node_attr_compose:
                Complex node features as (scalar, vector).
            edge_index_q_cps_knn:
                Query→complex KNN edges for building query context.
            atom_type_emb:
                Query atom-type embeddings.
            index_real_cps_edge_for_atten:
                Indices selecting the (i, j) pairs used by the edge attention.
            tri_edge_index, tri_edge_feat:
                Triangle/3-body features used to construct attention biases.
            annealing:
                If True, apply distance-based message annealing.

        Returns:
            z_latent, logabsdet_elementwise:
              - z_latent: latent representation, shape (E, B+1)
              - logabsdet_elementwise: elementwise log|∂z/∂x| contributions,
                shape (E, B+1). Sum over the last dimension for scalar log|det J|.

        Notes:
            If E == 0, returns empty tensors on the correct device/dtype.
        """
        if not _has_edges(edge_index_query):
            z_edge = torch.empty([0, self.num_bond_types + 1], device=pos_query.device, dtype=z_edge.dtype)
            edge_log_jacob = torch.empty(
                [0, self.num_bond_types + 1], device=pos_query.device, dtype=z_edge.dtype
            )
            return z_edge, edge_log_jacob

        y = self.pos_encoder(
            pos_query, edge_index_q_cps_knn, cpx_pos, node_attr_compose, atom_type_emb, annealing=annealing
        )
        idx_node_i = edge_index_query[0]
        node_mol_i: ScalarVectorFeatures = (y[0][idx_node_i], y[1][idx_node_i])
        idx_node_j = edge_index_query[1]
        node_mol_j: ScalarVectorFeatures = (
            node_attr_compose[0][idx_node_j],
            node_attr_compose[1][idx_node_j],
        )

        vec_ij = pos_query[idx_node_i] - cpx_pos[idx_node_j]
        dist_ij = torch.norm(vec_ij, p=2, dim=-1).view(-1, 1)  # [E, 1]

        edge_ij: ScalarVectorFeatures = (self.distance_expansion_3A(dist_ij), self.vector_expansion(vec_ij))
        edge_feat = self.nn_edge_ij(edge_ij)

        edge_attr: ScalarVectorFeatures = (
            torch.cat([node_mol_i[0], node_mol_j[0], edge_feat[0]], dim=-1),  # (E, F)
            torch.cat([node_mol_i[1], node_mol_j[1], edge_feat[1]], dim=1),
        )
        edge_attr = self.edge_feat(edge_attr)
        edge_attr = self.edge_atten(
            edge_attr,
            edge_index_query,
            cpx_pos,
            index_real_cps_edge_for_atten,
            tri_edge_index,
            tri_edge_feat,
        )

        # Apply normalizing flow layers
        edge_log_jacob = torch.zeros_like(z_edge)
        for flow_layer in self.flow_layers:
            log_s, t = flow_layer(edge_attr)
            z_edge = (z_edge + t) * torch.exp(log_s)
            edge_log_jacob += log_s

        return z_edge, edge_log_jacob

    def reverse(
        self,
        edge_latent: Tensor,
        pos_query: Tensor,
        edge_index_query: Tensor,
        cpx_pos: Tensor,
        node_attr_compose: ScalarVectorFeatures,
        edge_index_q_cps_knn: Tensor,
        atom_type_emb: Tensor,
        index_real_cps_edge_for_atten: tuple[Tensor, Tensor],
        tri_edge_index: tuple[Tensor, Tensor],
        tri_edge_feat: Tensor,
        annealing: bool = False,
    ) -> Tensor:
        """Map latent samples to dequantised edge/bond-type features.

        This applies the inverse of `forward` (latent→data) using the same edge
        conditioning features derived from geometry and complex context.

        Args:
            edge_latent:
                Latent samples in base space, shape (E, B+1).
            pos_query, edge_index_query, cpx_pos, node_attr_compose,
            edge_index_q_cps_knn, atom_type_emb,
            index_real_cps_edge_for_atten, tri_edge_index, tri_edge_feat, annealing:
                As in `forward`.

        Returns:
            Dequantised edge/bond-type features in data space, shape (E, B+1).

        Notes:
            If E == 0, returns an empty tensor on the correct device/dtype.
        """
        if not _has_edges(edge_index_query):
            edge_latent = torch.empty(
                [0, self.num_bond_types + 1], device=pos_query.device, dtype=edge_latent.dtype
            )
            return edge_latent

        y = self.pos_encoder(
            pos_query, edge_index_q_cps_knn, cpx_pos, node_attr_compose, atom_type_emb, annealing=annealing
        )
        idx_node_i = edge_index_query[0]
        node_mol_i: ScalarVectorFeatures = (y[0][idx_node_i], y[1][idx_node_i])
        idx_node_j = edge_index_query[1]
        node_mol_j: ScalarVectorFeatures = (
            node_attr_compose[0][idx_node_j],
            node_attr_compose[1][idx_node_j],
        )

        vec_ij = pos_query[idx_node_i] - cpx_pos[idx_node_j]
        dist_ij = torch.norm(vec_ij, p=2, dim=-1).view(-1, 1)  # [E, 1]

        edge_ij: ScalarVectorFeatures = (self.distance_expansion_3A(dist_ij), self.vector_expansion(vec_ij))
        edge_feat = self.nn_edge_ij(edge_ij)  # (E, F)

        edge_attr: ScalarVectorFeatures = (
            torch.cat([node_mol_i[0], node_mol_j[0], edge_feat[0]], dim=-1),  # (E, F)
            torch.cat([node_mol_i[1], node_mol_j[1], edge_feat[1]], dim=1),
        )
        edge_attr = self.edge_feat(edge_attr)
        edge_attr = self.edge_atten(
            edge_attr,
            edge_index_query,
            cpx_pos,
            index_real_cps_edge_for_atten,
            tri_edge_index,
            tri_edge_feat,
        )

        # Apply inverse flow layers (in reverse order)
        for flow_layer in reversed(self.flow_layers):
            log_s, t = flow_layer(edge_attr)
            edge_latent = (edge_latent / torch.exp(log_s)) - t

        return edge_latent
