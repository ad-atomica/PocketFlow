"""Top-level PocketFlow model and training loss.

This module defines :class:`~pocket_flow.gdbp_model.pocket_flow.PocketFlow`, the
core model used by PocketFlow's *GDBP* (geometry-dependent belief propagation)
stack. In training, the model consumes a "composed" protein+ligand-context graph
and predicts (at each ligand-growing step):

- **Focal/frontier logits** for candidate attachment sites in the ligand context
  (and optionally on the apo protein surface),
- **Next-atom type** via a conditional normalising flow over dequantised
  categorical atom-type features,
- **Next-atom 3D position** via a mixture-density network (MDN),
- **Bond types** from the new atom to nearby context atoms via a conditional
  normalising flow.

The main entrypoint is :meth:`PocketFlow.get_loss`, which returns a dictionary
containing the total loss and individual loss terms.

Notes:
    - This file also contains a small set of module-level `EasyDict` configs
      (`encoder_cfg`, `focal_net_cfg`, ...) intended as convenient defaults /
      examples. Training scripts in the repo typically construct a richer config
      object and pass it to :class:`PocketFlow`.
    - All geometric networks in this package use paired scalar/vector features:
      scalar features have shape ``(N, F_sca)`` and vector features have shape
      ``(N, F_vec, 3)``.
"""

from __future__ import annotations

from typing import Any, Protocol, override

import torch
import torch.nn.functional as F
from easydict import EasyDict
from torch import Tensor, nn

from pocket_flow.gdbp_model.atom_flow import AtomFlow
from pocket_flow.gdbp_model.bond_flow import BondFlow
from pocket_flow.gdbp_model.encoder import ContextEncoder
from pocket_flow.gdbp_model.focal_net import FocalNet
from pocket_flow.gdbp_model.net_utils import AtomEmbedding, embed_compose
from pocket_flow.gdbp_model.position_predictor import PositionPredictor
from pocket_flow.gdbp_model.types import ScalarVectorFeatures

EPS_PROBABILITY: float = 1e-16


class TrainingBatch(Protocol):
    """Structural protocol for the training-time batch expected by `PocketFlow`.

    Instances are typically `torch_geometric.data.Data`/`Batch` objects produced
    by the transforms in `pocket_flow/utils/transform.py` and `collate_fn`.

    Most tensors follow a small set of indexing conventions:
      - **Composed/complex tensors** (`cpx_*`) are indexed in the concatenated
        space of ligand-context atoms followed by protein atoms.
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
            Edge indices for the composed context graph, shape ``(2, E_cpx)``.
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
        focal_idx_in_context:
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
            kNN edges from query positions to composed nodes, analogous to
            `edge_query_index_*` but used to build conditioning context for edge
            prediction, each with shape ``(E_knn,)``.
        index_real_cps_edge_for_atten:
            Pair indices for edge-attention over the queried edges, stored as a
            stacked tensor of shape ``(2, E_att)`` and split into a tuple before
            passing to `BondFlow`.
        tri_edge_index:
            Pair indices (in composed node space) defining "triangle edges" used
            by edge-attention, stored as a stacked tensor of shape
            ``(2, E_att)``.
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
    focal_idx_in_context: Tensor
    y_pos: Tensor
    edge_label: Tensor
    edge_query_index_0: Tensor
    edge_query_index_1: Tensor
    pos_query_knn_edge_idx_0: Tensor
    pos_query_knn_edge_idx_1: Tensor
    index_real_cps_edge_for_atten: Tensor
    tri_edge_index: Tensor
    tri_edge_feat: Tensor


encoder_cfg: EasyDict = EasyDict({"edge_channels": 8, "num_interactions": 6, "knn": 32, "cutoff": 10.0})
focal_net_cfg: EasyDict = EasyDict({"hidden_dim_sca": 32, "hidden_dim_vec": 8})
atom_flow_cfg: EasyDict = EasyDict({"hidden_dim_sca": 32, "hidden_dim_vec": 8, "num_flow_layers": 6})
pos_predictor_cfg: EasyDict = EasyDict({"num_filters": [64, 64], "n_component": 3})
pos_filter_cfg: EasyDict = EasyDict({"edge_channels": 8, "num_filters": [32, 16]})
edge_flow_cfg: EasyDict = EasyDict(
    {
        "edge_channels": 8,
        "num_filters": [32, 8],
        "num_bond_types": 3,
        "num_heads": 2,
        "cutoff": 10.0,
        "num_flow_layers": 3,
    }
)
config: EasyDict = EasyDict(
    {
        "deq_coeff": 0.9,
        "hidden_channels": 32,
        "hidden_channels_vec": 8,
        "bottleneck": 8,
        "use_conv1d": False,
        "encoder": encoder_cfg,
        "atom_flow": atom_flow_cfg,
        "pos_predictor": pos_predictor_cfg,
        "pos_filter": pos_filter_cfg,
        "edge_flow": edge_flow_cfg,
        "focal_net": focal_net_cfg,
    }
)


class PocketFlow(nn.Module):
    """PocketFlow GDBP model with focal head, MDN, and conditional flows.

    Components:
        - `AtomEmbedding` modules for ligand-context atoms and protein atoms
          (`protein_atom_emb`, `ligand_atom_emb`) to lift raw features into the
          shared scalar/vector representation.
        - `ContextEncoder` to encode composed context features on the composed
          graph (`cpx_edge_index`, `cpx_edge_feature`) with geometric message
          passing.
        - `FocalNet` to produce a binary logit per selected node (frontier/focal
          supervision).
        - `AtomFlow` to model the distribution over next-atom types conditioned
          on encoded features at the focal atom(s).
        - `PositionPredictor` (MDN) to model the distribution over next-atom 3D
          positions conditioned on the focal atom and the chosen atom type.
        - `BondFlow` to model the distribution over bond types from the new atom
          to nearby context atoms conditioned on geometry and context features.

    Training objective:
        :meth:`get_loss` computes and returns a dictionary with:
          - `loss`: sum of all terms (with `nan_to_num` safety)
          - `loss_atom`: atom-type negative log-likelihood term
          - `loss_pos`: position negative log-likelihood term (MDN)
          - `loss_edge`: bond-type negative log-likelihood term
          - `focal_loss`: ligand-context frontier BCE term
          - `surf_loss`: apo protein surface BCE term

    The module's `forward` simply calls :meth:`get_loss` (so this model is
    "loss-first" during training).
    """

    config: Any
    num_bond_types: int
    msg_annealing: bool
    emb_dim: tuple[int, int]
    protein_atom_emb: AtomEmbedding
    ligand_atom_emb: AtomEmbedding
    atom_type_embedding: nn.Embedding
    encoder: ContextEncoder
    focal_net: FocalNet
    atom_flow: AtomFlow
    pos_predictor: PositionPredictor
    edge_flow: BondFlow

    def __init__(self, config: Any) -> None:
        """Construct a PocketFlow model.

        The provided config is expected to be an object with attribute access
        (commonly `easydict.EasyDict`) containing, at minimum:

        - `hidden_channels`, `hidden_channels_vec`: scalar/vector widths
        - `protein_atom_feature_dim`, `ligand_atom_feature_dim`: input dims
        - `num_atom_type`: number of atom-type classes for `AtomFlow`
        - `num_bond_types`: number of bond-type classes used by the encoder
        - `msg_annealing`: whether to enable distance-based message annealing
        - Sub-configs: `encoder`, `focal_net`, `atom_flow`, `pos_predictor`,
          `edge_flow`

        Args:
            config: Configuration object (attribute-accessible).
        """
        super().__init__()
        self.config = config
        self.num_bond_types = config.num_bond_types
        self.msg_annealing = config.msg_annealing
        self.emb_dim = (config.hidden_channels, config.hidden_channels_vec)
        self.protein_atom_emb = AtomEmbedding(config.protein_atom_feature_dim, 1, *self.emb_dim)
        self.ligand_atom_emb = AtomEmbedding(config.ligand_atom_feature_dim, 1, *self.emb_dim)
        self.atom_type_embedding = nn.Embedding(config.num_atom_type, config.hidden_channels)

        self.encoder = ContextEncoder(
            hidden_channels=self.emb_dim,
            edge_channels=config.encoder.edge_channels,
            num_edge_types=config.num_bond_types,
            num_interactions=config.encoder.num_interactions,
            k=config.encoder.knn,
            cutoff=config.encoder.cutoff,
            bottleneck=config.bottleneck,
            use_conv1d=config.use_conv1d,
            num_heads=config.encoder.num_heads,
        )
        self.focal_net = FocalNet(
            self.emb_dim[0],
            self.emb_dim[1],
            config.focal_net.hidden_dim_sca,
            config.focal_net.hidden_dim_vec,
            bottleneck=config.bottleneck,
            use_conv1d=config.use_conv1d,
        )
        self.atom_flow = AtomFlow(
            self.emb_dim[0],
            self.emb_dim[1],
            config.atom_flow.hidden_dim_sca,
            config.atom_flow.hidden_dim_vec,
            num_lig_atom_type=config.num_atom_type,
            num_flow_layers=config.atom_flow.num_flow_layers,
            bottleneck=config.bottleneck,
            use_conv1d=config.use_conv1d,
        )
        self.pos_predictor = PositionPredictor(
            self.emb_dim[0],
            self.emb_dim[1],
            config.pos_predictor.num_filters,
            config.pos_predictor.n_component,
            bottleneck=config.bottleneck,
            use_conv1d=config.use_conv1d,
        )

        self.edge_flow = BondFlow(
            self.emb_dim[0],
            self.emb_dim[1],
            config.edge_flow.edge_channels,
            config.edge_flow.num_filters,
            config.edge_flow.num_bond_types,
            num_heads=config.edge_flow.num_heads,
            cutoff=config.edge_flow.cutoff,
            num_st_layers=config.edge_flow.num_flow_layers,
            bottleneck=config.bottleneck,
            use_conv1d=config.use_conv1d,
        )

    def get_parameter_number(self) -> dict[str, int]:
        """Return total and trainable parameter counts."""
        total_num: int = sum(p.numel() for p in self.parameters())
        trainable_num: int = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"Total": total_num, "Trainable": trainable_num}

    @override
    def forward(self, data: TrainingBatch) -> dict[str, Tensor]:
        """Alias for :meth:`get_loss` to support `nn.Module` calling convention."""
        return self.get_loss(data)

    def get_loss(self, data: TrainingBatch) -> dict[str, Tensor]:
        """Compute training losses for a batch.

        High-level flow:
            1) Embed ligand-context/protein atoms into scalar/vector features and
               encode them with the `ContextEncoder`.
            2) Compute focal/frontier supervision with `FocalNet`:
               - `focal_loss` on ligand-context nodes
               - `surf_loss` on apo protein nodes (may be empty)
            3) Compute atom-type negative log-likelihood via `AtomFlow` on a
               *dequantised* one-hot representation of `atom_label`.
            4) Compute position negative log-likelihood under the MDN predicted
               by `PositionPredictor`, conditioned on the focal atom and the
               atom-type embedding.
            5) Compute bond-type negative log-likelihood via `BondFlow` on a
               dequantised one-hot representation of `edge_label`.

        Dequantisation:
            Categorical labels are converted to one-hot vectors and perturbed by
            uniform noise: `x <- one_hot(label) + deq_coeff * U(0, 1)`. This
            makes the flow objective well-defined on continuous inputs.

        Args:
            data: Batch object satisfying :class:`TrainingBatch`.

        Returns:
            Dictionary with keys:
              - `loss`: summed loss (scalar)
              - `loss_atom`, `loss_pos`, `loss_edge`: NLL terms (scalars)
              - `focal_loss`, `surf_loss`: BCE terms (scalars)
        """
        h_cpx_list: list[Tensor] = embed_compose(
            data.cpx_feature.float(),
            data.cpx_pos,
            data.idx_ligand_ctx_in_cpx,
            data.idx_protein_in_cpx,
            self.ligand_atom_emb,
            self.protein_atom_emb,
            self.emb_dim,
        )
        h_cpx: ScalarVectorFeatures = (h_cpx_list[0], h_cpx_list[1])

        # encoding context
        h_cpx = self.encoder(
            node_attr=h_cpx,
            pos=data.cpx_pos,
            edge_index=data.cpx_edge_index,
            edge_feature=data.cpx_edge_feature,
            annealing=self.msg_annealing,
        )
        # for focal loss
        focal_pred: Tensor = self.focal_net(h_cpx, data.idx_ligand_ctx_in_cpx)
        focal_loss: Tensor = F.binary_cross_entropy_with_logits(
            input=focal_pred, target=data.ligand_frontier.view(-1, 1).float()
        )
        # for focal loss in protein
        focal_pred_apo: Tensor = self.focal_net(h_cpx, data.apo_protein_idx)
        surf_loss: Tensor = F.binary_cross_entropy_with_logits(
            input=focal_pred_apo, target=data.candidate_focal_label_in_protein.view(-1, 1).float()
        )
        # for atom loss
        x_z: Tensor = F.one_hot(data.atom_label, num_classes=self.config.num_atom_type).float()
        x_z = x_z + self.config.deq_coeff * torch.rand(x_z.size(), device=x_z.device)
        z_atom, atom_log_jacob = self.atom_flow(x_z, h_cpx, data.focal_idx_in_context)
        ll_atom: Tensor = (0.5 * (z_atom**2) - atom_log_jacob).mean()

        # for position loss
        atom_type_emb: Tensor = self.atom_type_embedding(data.atom_label)
        _, abs_mu, sigma, pi = self.pos_predictor(
            list(h_cpx),
            data.focal_idx_in_context,
            data.cpx_pos,
            atom_type_emb=atom_type_emb,
        )
        loss_pos: Tensor = -torch.log(
            self.pos_predictor.get_mdn_probability(abs_mu, sigma, pi, data.y_pos).clamp_min(EPS_PROBABILITY)
        ).mean()

        # for edge loss
        z_edge: Tensor = F.one_hot(data.edge_label, num_classes=4).float()
        z_edge = z_edge + self.config.deq_coeff * torch.rand(z_edge.size(), device=z_edge.device)
        edge_index_query: Tensor = torch.stack([data.edge_query_index_0, data.edge_query_index_1])
        pos_query_knn_edge_idx: Tensor = torch.stack(
            [data.pos_query_knn_edge_idx_0, data.pos_query_knn_edge_idx_1]
        )
        z_edge, edge_log_jacob = self.edge_flow(
            z_edge=z_edge,
            pos_query=data.y_pos,
            edge_index_query=edge_index_query,
            cpx_pos=data.cpx_pos,
            node_attr_compose=h_cpx,
            edge_index_q_cps_knn=pos_query_knn_edge_idx,
            index_real_cps_edge_for_atten=(
                data.index_real_cps_edge_for_atten[0],
                data.index_real_cps_edge_for_atten[1],
            ),
            tri_edge_index=(data.tri_edge_index[0], data.tri_edge_index[1]),
            tri_edge_feat=data.tri_edge_feat,
            atom_type_emb=atom_type_emb,
            annealing=self.msg_annealing,
        )
        # loss all
        ll_edge: Tensor = (0.5 * (z_edge**2) - edge_log_jacob).mean()

        loss: Tensor = (
            torch.nan_to_num(ll_atom)
            + torch.nan_to_num(loss_pos)
            + torch.nan_to_num(ll_edge)
            + torch.nan_to_num(focal_loss)
            + torch.nan_to_num(surf_loss)
        )
        return {
            "loss": loss,
            "loss_atom": ll_atom,
            "loss_edge": ll_edge,
            "loss_pos": loss_pos,
            "focal_loss": focal_loss,
            "surf_loss": torch.nan_to_num(surf_loss),
        }
