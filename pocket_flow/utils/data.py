"""
PyG `Data` helpers for protein–ligand complex examples.

This module defines the core data container used throughout PocketFlow training
and generation: `ComplexData`, a `torch_geometric.data.Data` subclass that
stores protein atoms, ligand atoms, and several derived “context” / “complex”
graph views used by downstream transforms and models.

The codebase follows a loose naming convention:

- Protein fields are prefixed with `protein_` (e.g., `protein_pos`).
- Ligand fields are prefixed with `ligand_` (e.g., `ligand_bond_index`).
- “Context” fields describe the subset of ligand atoms currently considered as
  the growth context (e.g., `context_idx`, `ligand_context_pos`).
- “Complex” / `cpx_` fields describe a combined graph containing both protein
  and ligand-context atoms (e.g., `cpx_pos`, `cpx_edge_index`).
- `idx_ligand_ctx_in_cpx` maps ligand-context indices into the composed
  `cpx_*` index space and should be used whenever ctx indices are converted to
  composed indices.

The batching behavior for many index-like fields is customized via
`ComplexData.__inc__` so that `torch_geometric.data.Batch.from_data_list` shifts
indices correctly when concatenating multiple examples.
"""

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data
from torch_geometric.data.data import BaseData

FOLLOW_BATCH: list[str] = []  # ['protein_element', 'ligand_context_element', 'pos_real', 'pos_fake']

RingInfoVector = NDArray[np.int64]
LigandRingInfo = dict[int, RingInfoVector]
NeighborList = dict[int, list[int]]


class ComplexData(Data):
    """
    `torch_geometric.data.Data` representing a protein–ligand complex example.

    This type is the primary “sample” object passed between dataset code,
    transforms, and the model. It is intentionally flexible: most attributes are
    created dynamically (by setting fields on the `Data` store), but this class
    declares many commonly used fields for type checking and editor support.

    A few notable conventions/expectations used across the repo:

    - Atom elements are stored as atomic numbers in `*_element` (e.g., 6 for C).
    - Coordinates are stored in `*_pos` as float tensors of shape `(N, 3)`.
    - Edge indices follow PyG conventions: shape `(2, E)` with `dtype=torch.long`.
    - When `is_traj` is `True`, the object is treated as a trajectory-style
      complex graph and `num_nodes` maps to `cpx_pos.size(0)`.

    Attributes are not exhaustive; transforms may attach additional fields.
    """

    protein_element: Tensor
    protein_pos: Tensor
    protein_is_backbone: Tensor
    protein_atom_name: list[str]
    protein_atom_to_aa_type: Tensor
    protein_atom_feature: Tensor
    protein_bond_index: Tensor
    protein_bond_type: Tensor
    protein_surface_mask: Tensor | None
    protein_molecule_name: str | None
    protein_filename: str

    ligand_element: Tensor
    ligand_pos: Tensor
    ligand_bond_index: Tensor
    ligand_bond_type: Tensor
    ligand_center_of_mass: Tensor
    ligand_atom_feature: Tensor
    ligand_atom_feature_full: Tensor
    ligand_atom_num_bonds: Tensor
    ligand_atom_valence: Tensor
    ligand_num_neighbors: Tensor
    ligand_ring_info: LigandRingInfo
    ligand_nbh_list: NeighborList
    ligand_filename: str

    context_idx: Tensor
    masked_idx: Tensor
    ligand_masked_element: Tensor
    ligand_masked_pos: Tensor
    ligand_context_element: Tensor
    ligand_context_pos: Tensor
    ligand_context_feature_full: Tensor
    ligand_context_bond_index: Tensor
    ligand_context_bond_type: Tensor
    ligand_context_num_neighbors: Tensor
    ligand_context_valence: Tensor
    ligand_context_num_bonds: Tensor
    ligand_frontier: Tensor
    max_atom_valence: Tensor

    cpx_pos: Tensor
    cpx_feature: Tensor
    cpx_edge_index: Tensor
    cpx_edge_type: Tensor
    cpx_edge_feature: Tensor
    cpx_knn_edge_index: Tensor
    idx_ligand_ctx_in_cpx: Tensor
    idx_protein_in_cpx: Tensor
    cpx_backbone_index: Tensor
    step_batch: Tensor

    focal_idx_in_cpx: Tensor
    focal_idx_in_context_selected: Tensor
    focal_idx_in_context_candidates: Tensor
    focal_label: Tensor
    atom_label: Tensor
    edge_label: Tensor
    edge_query_index_0: Tensor
    edge_query_index_1: Tensor
    index_real_cps_edge_for_atten: Tensor
    tri_edge_index: Tensor
    tri_edge_feat: Tensor
    candidate_focal_idx_in_protein: Tensor
    candidate_focal_label_in_protein: Tensor
    apo_protein_idx: Tensor

    y_pos: Tensor
    pos_query_knn_edge_idx_0: Tensor
    pos_query_knn_edge_idx_1: Tensor
    pos_fake: Tensor
    pos_fake_knn_edge_idx_0: Tensor
    pos_fake_knn_edge_idx_1: Tensor
    pos_real: Tensor
    pos_real_knn_edge_idx_0: Tensor
    pos_real_knn_edge_idx_1: Tensor

    cpx_pos_batch: Tensor
    y_pos_batch: Tensor
    edge_label_batch: Tensor
    atom_label_batch: Tensor

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Create a `ComplexData` instance.

        Args:
            *args: Forwarded to `torch_geometric.data.Data`.
            **kwargs: Forwarded to `torch_geometric.data.Data`.

        Notes:
            The instance starts with `is_traj = False`. Some code paths (e.g.
            trajectory datasets or generation-time rollouts) flip this flag to
            change `num_nodes` semantics.
        """
        super().__init__(*args, **kwargs)
        self.is_traj: bool = False

    @staticmethod
    def from_protein_ligand_dicts(
        protein_dict: Mapping[str, Any] | None = None,
        ligand_dict: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> "ComplexData":
        """
        Construct a `ComplexData` from prefixed protein/ligand dictionaries.

        Each key in `protein_dict` is stored on the returned instance under the
        field name `protein_{key}`; similarly `ligand_dict` maps to `ligand_{key}`.

        The helper also derives `ligand_nbh_list` (a Python adjacency list) from
        `ligand_bond_index`. This is used by some transforms for convenient
        neighbor lookup without materializing sparse tensors.

        Args:
            protein_dict: Mapping of unprefixed protein fields to store
                (e.g. `{"pos": Tensor, "element": Tensor}`).
            ligand_dict: Mapping of unprefixed ligand fields to store.
            **kwargs: Additional fields forwarded to the constructor.

        Returns:
            A new `ComplexData` instance with provided fields installed.

        Raises:
            AttributeError: If `ligand_bond_index` is not present after applying
                `ligand_dict`/`kwargs` (required to build `ligand_nbh_list`).
        """
        instance = ComplexData(**kwargs)

        if protein_dict is not None:
            for key, item in protein_dict.items():
                instance["protein_" + key] = item

        if ligand_dict is not None:
            for key, item in ligand_dict.items():
                instance["ligand_" + key] = item

        instance["ligand_nbh_list"] = {
            i.item(): [
                j.item()
                for k, j in enumerate(instance.ligand_bond_index[1])
                if instance.ligand_bond_index[0, k].item() == i
            ]
            for i in instance.ligand_bond_index[0]
        }
        return instance

    def __inc__(self, key: str, value: Any, *args: Any, **kwargs: Any) -> int | torch.Tensor | torch.Size:
        """
        Return the per-example increment used by PyG when batching `key`.

        PyG uses `Data.__inc__` to determine how to shift index tensors (such as
        node indices in `edge_index`) when concatenating multiple examples into a
        `Batch`. This override provides correct offsets for a set of custom
        index-like fields that refer to different node sets:

        - Fields indexing `cpx_pos` are incremented by `cpx_pos.size(0)`.
        - Fields indexing `y_pos`, `pos_fake`, or `pos_real` use their respective
          first dimensions.
        - `step_batch` increments by `(step_batch.max() + 1)` to avoid collisions.

        Args:
            key: Field name being batched.
            value: Field value for this example (unused; included for signature
                compatibility with PyG).
            *args: Ignored extra arguments passed by PyG.
            **kwargs: Ignored extra keyword arguments passed by PyG.

        Returns:
            The increment used to offset `key` during batching.
        """
        keys: set[str] = {
            "idx_ligand_ctx_in_cpx",
            "idx_protein_in_cpx",
            "focal_idx_in_cpx",
            "focal_idx_in_context_candidates",
            "focal_idx_in_context_selected",
            "cpx_knn_edge_index",
            "edge_query_index_1",
            "pos_query_knn_edge_idx_0",
            "pos_fake_knn_edge_idx_0",
            "pos_real_knn_edge_idx_0",
            "tri_edge_index",
            "apo_protein_idx",
            "candidate_focal_idx_in_protein",
            "cpx_backbone_index",
        }
        if key in keys:
            return self["cpx_pos"].size(0)
        elif key == "edge_query_index_0":
            return self["y_pos"].size(0)
        elif key == "pos_query_knn_edge_idx_1":
            return self["y_pos"].size(0)
        elif key == "pos_fake_knn_edge_idx_1":
            return self["pos_fake"].size(0)
        elif key == "pos_real_knn_edge_idx_1":
            return self["pos_real"].size(0)
        elif key == "step_batch":
            return self["step_batch"].max() + 1
        elif key == "ligand_bond_index":
            return self["ligand_element"].size()
        elif key == "index_real_cps_edge_for_atten":
            return self["edge_query_index_0"].size(0)
        else:
            return super().__inc__(key, value)

    @property
    def num_nodes(self) -> int:
        """
        Number of nodes for PyG bookkeeping.

        Returns:
            If `is_traj` is `True`, returns `cpx_pos.size(0)`. Otherwise returns
            the combined size of protein atoms and ligand-context atoms:
            `protein_pos.size(0) + context_idx.size(0)`.
        """
        if self.is_traj:
            return self.cpx_pos.size(0)
        else:
            return self.protein_pos.size(0) + self.context_idx.size(0)


class ComplexDataTrajectory(ComplexData):
    """
    `ComplexData` representing a trajectory-collated complex.

    Unlike a single-step `ComplexData` sample, a `ComplexDataTrajectory` is a
    *concatenation* of multiple growth steps for the same underlying complex.
    Tensors such as `cpx_pos` are concatenated across steps, and a per-node
    step assignment vector `step_batch` indicates which step each row belongs
    to.

    This is intentionally distinct from `torch_geometric.data.Batch`: it
    represents multiple steps of a single complex, not multiple independent
    examples.

    Required fields:
      - `cpx_pos`: `(N, 3)` complex node positions across all steps.
      - `step_batch`: `(N,)` long tensor mapping each `cpx_pos` row to a step id
        in `0..num_steps-1`.
    """

    step_batch: Tensor
    cpx_pos_batch: Tensor
    y_pos_batch: Tensor
    edge_label_batch: Tensor
    atom_label_batch: Tensor

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.is_traj = True  # tODO: Do we need this now we have the ComplexDataTrajectory class ?

    @property
    def num_steps(self) -> int:
        step_batch = getattr(self, "step_batch", None)
        if step_batch is None or not isinstance(step_batch, torch.Tensor) or step_batch.numel() == 0:
            return 0
        return int(step_batch.max().item()) + 1

    def validate(self, *, strict: bool = True) -> None:
        if strict and not getattr(self, "is_traj", False):
            raise ValueError("ComplexDataTrajectory requires `is_traj == True`.")

        if not hasattr(self, "step_batch"):
            raise ValueError("ComplexDataTrajectory requires `step_batch`.")
        if not isinstance(self.step_batch, torch.Tensor) or self.step_batch.ndim != 1:
            raise ValueError("ComplexDataTrajectory requires `step_batch` to be a 1D tensor.")
        if strict and self.step_batch.dtype != torch.long:
            raise ValueError("ComplexDataTrajectory requires `step_batch` to have dtype `torch.long`.")

        if not hasattr(self, "cpx_pos"):
            raise ValueError("ComplexDataTrajectory requires `cpx_pos`.")
        if self.step_batch.numel() != self.cpx_pos.size(0):
            raise ValueError(
                "ComplexDataTrajectory requires `step_batch.numel() == cpx_pos.size(0)` "
                f"(got {self.step_batch.numel()} vs {self.cpx_pos.size(0)})."
            )

        if self.step_batch.numel() > 0:
            if int(self.step_batch.min().item()) < 0:
                raise ValueError("ComplexDataTrajectory requires `step_batch` to be non-negative.")
            num_steps = self.num_steps
            if int(self.step_batch.max().item()) >= num_steps:
                raise ValueError("ComplexDataTrajectory requires `step_batch` values < `num_steps`.")

        if hasattr(self, "cpx_pos_batch"):
            if not isinstance(self.cpx_pos_batch, torch.Tensor) or self.cpx_pos_batch.ndim != 1:
                raise ValueError(
                    "ComplexDataTrajectory requires `cpx_pos_batch` to be a 1D tensor when present."
                )
            if self.cpx_pos_batch.numel() != self.cpx_pos.size(0):
                raise ValueError("ComplexDataTrajectory requires `cpx_pos_batch.numel() == cpx_pos.size(0)`.")
            if strict and not torch.equal(self.cpx_pos_batch, self.step_batch):
                raise ValueError(
                    "ComplexDataTrajectory requires `cpx_pos_batch` to equal `step_batch` when present."
                )

        def _validate_batch(
            *,
            batch_name: str,
            tensor_name: str,
        ) -> None:
            if not hasattr(self, batch_name):
                return
            batch = getattr(self, batch_name)
            if not isinstance(batch, torch.Tensor) or batch.ndim != 1:
                raise ValueError(
                    f"ComplexDataTrajectory requires `{batch_name}` to be a 1D tensor when present."
                )
            if strict and batch.dtype != torch.long:
                raise ValueError(f"ComplexDataTrajectory requires `{batch_name}` to have dtype `torch.long`.")

            if not hasattr(self, tensor_name):
                raise ValueError(f"ComplexDataTrajectory has `{batch_name}` but is missing `{tensor_name}`.")
            tensor = getattr(self, tensor_name)
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"ComplexDataTrajectory requires `{tensor_name}` to be a tensor.")
            if batch.numel() != tensor.size(0):
                raise ValueError(
                    f"ComplexDataTrajectory requires `{batch_name}.numel() == {tensor_name}.size(0)` "
                    f"(got {batch.numel()} vs {tensor.size(0)})."
                )
            if batch.numel() > 0:
                if int(batch.min().item()) < 0:
                    raise ValueError(f"ComplexDataTrajectory requires `{batch_name}` to be non-negative.")
                if int(batch.max().item()) >= self.num_steps:
                    raise ValueError(f"ComplexDataTrajectory requires `{batch_name}` values < `num_steps`.")

        _validate_batch(batch_name="y_pos_batch", tensor_name="y_pos")
        _validate_batch(batch_name="edge_label_batch", tensor_name="edge_label")
        _validate_batch(batch_name="atom_label_batch", tensor_name="atom_label")

    @classmethod
    def from_steps(
        cls,
        steps: Sequence["ComplexData"],
        collate_keys: Sequence[str] | None = None,
        *,
        validate: bool = True,
    ) -> "ComplexDataTrajectory":
        """
        Collate a list of trajectory steps into a single `ComplexDataTrajectory`.

        The function concatenates tensor fields listed in `collate_keys`, offsets
        index-like fields so they remain valid after concatenation, and attaches
        assignment vectors that map rows back to the originating step.
        """
        if len(steps) == 0:
            raise ValueError("ComplexDataTrajectory.from_steps requires a non-empty `steps`.")

        data_dict: dict[str, Any]
        required_collate_keys = {
            "step_batch",
            "cpx_pos_batch",
            "y_pos_batch",
            "edge_label_batch",
            "atom_label_batch",
        }
        if collate_keys:
            keys = list(dict.fromkeys([*collate_keys, *sorted(required_collate_keys)]))
            data_dict = {k: [] for k in keys}
        else:
            keys = list(dict.fromkeys([*list(steps[0].keys()), *sorted(required_collate_keys)]))
            data_dict = {k: [] for k in keys}

        data_dict["protein_pos"] = steps[0].protein_pos
        data_dict["protein_atom_feature"] = steps[0].protein_atom_feature
        data_dict["ligand_pos"] = steps[0].ligand_pos
        data_dict["ligand_element"] = steps[0].ligand_element
        data_dict["ligand_bond_index"] = steps[0].ligand_bond_index
        data_dict["ligand_bond_type"] = steps[0].ligand_bond_type
        data_dict["ligand_atom_feature_full"] = steps[0].ligand_atom_feature_full

        base_fields: tuple[str, ...] = (
            "protein_pos",
            "protein_atom_feature",
            "ligand_pos",
            "ligand_element",
            "ligand_bond_index",
            "ligand_bond_type",
            "ligand_atom_feature_full",
        )
        for step_idx, step in enumerate(steps[1:], start=1):
            for field in base_fields:
                if not hasattr(step, field):
                    raise ValueError(
                        "ComplexDataTrajectory.from_steps requires "
                        f"`{field}` on every step (missing on step {step_idx})."
                    )
                a = getattr(steps[0], field)
                b = getattr(step, field)
                if isinstance(a, torch.Tensor):
                    if not isinstance(b, torch.Tensor) or a.shape != b.shape or a.dtype != b.dtype:
                        raise ValueError(
                            "ComplexDataTrajectory.from_steps requires "
                            f"`{field}` to be identical across steps (mismatch on step {step_idx})."
                        )
                    if a.device != b.device:
                        raise ValueError(
                            "ComplexDataTrajectory.from_steps requires "
                            f"`{field}` to be on the same device across steps (mismatch on step {step_idx})."
                        )
                    if not torch.equal(a, b):
                        raise ValueError(
                            "ComplexDataTrajectory.from_steps requires "
                            f"`{field}` to be identical across steps (mismatch on step {step_idx})."
                        )
                else:
                    if a != b:
                        raise ValueError(
                            "ComplexDataTrajectory.from_steps requires "
                            f"`{field}` to be identical across steps (mismatch on step {step_idx})."
                        )

        compose_pos_cusum = 0
        edge_query_index_cusum = 0
        pos_fake_cusum = 0
        pos_real_cusum = 0

        for idx, d in enumerate(steps):
            data_dict["cpx_pos"].append(d["cpx_pos"])
            data_dict["cpx_feature"].append(d["cpx_feature"])
            data_dict["idx_ligand_ctx_in_cpx"].append(d.idx_ligand_ctx_in_cpx + compose_pos_cusum)
            data_dict["idx_protein_in_cpx"].append(d.idx_protein_in_cpx + compose_pos_cusum)
            data_dict["candidate_focal_idx_in_protein"].append(d.candidate_focal_idx_in_protein)
            data_dict["candidate_focal_label_in_protein"].append(d.candidate_focal_label_in_protein)
            data_dict["apo_protein_idx"].append(d.apo_protein_idx)
            data_dict["focal_idx_in_context_selected"].append(
                d.focal_idx_in_context_selected + compose_pos_cusum
            )
            data_dict["focal_idx_in_context_candidates"].append(
                d.focal_idx_in_context_candidates + compose_pos_cusum
            )
            data_dict["cpx_edge_index"].append(d.cpx_edge_index + compose_pos_cusum)
            data_dict["cpx_edge_type"].append(d.cpx_edge_type)
            data_dict["cpx_edge_feature"].append(d.cpx_edge_feature)
            data_dict["cpx_backbone_index"].append(d.cpx_backbone_index + compose_pos_cusum)
            data_dict["focal_label"].append(d.focal_label)
            data_dict["y_pos"].append(d.y_pos)
            data_dict["ligand_frontier"].append(d.ligand_frontier)
            data_dict["edge_label"].append(d.edge_label)
            data_dict["atom_label"].append(d.atom_label)
            data_dict["edge_query_index_0"].append(d.edge_query_index_0 + idx)
            data_dict["edge_query_index_1"].append(d.edge_query_index_1 + compose_pos_cusum)
            data_dict["pos_query_knn_edge_idx_0"].append(d.pos_query_knn_edge_idx_0 + compose_pos_cusum)
            data_dict["pos_query_knn_edge_idx_1"].append(d.pos_query_knn_edge_idx_1 + idx)
            data_dict["pos_fake"].append(d.pos_fake)
            data_dict["pos_fake_knn_edge_idx_0"].append(d.pos_fake_knn_edge_idx_0 + compose_pos_cusum)
            data_dict["pos_fake_knn_edge_idx_1"].append(d.pos_fake_knn_edge_idx_1 + pos_fake_cusum)
            data_dict["pos_real"].append(d.pos_real)
            data_dict["pos_real_knn_edge_idx_0"].append(d.pos_real_knn_edge_idx_0 + compose_pos_cusum)
            data_dict["pos_real_knn_edge_idx_1"].append(d.pos_real_knn_edge_idx_1 + pos_real_cusum)
            data_dict["index_real_cps_edge_for_atten"].append(
                d.index_real_cps_edge_for_atten + edge_query_index_cusum
            )
            data_dict["tri_edge_index"].append(d.tri_edge_index + compose_pos_cusum)
            data_dict["tri_edge_feat"].append(d.tri_edge_feat)
            data_dict["step_batch"].append(
                torch.full(
                    (d.cpx_pos.size(0),),
                    idx,
                    dtype=torch.long,
                    device=d.cpx_pos.device,
                )
            )
            data_dict["y_pos_batch"].append(
                torch.full(
                    (d.y_pos.size(0),),
                    idx,
                    dtype=torch.long,
                    device=d.y_pos.device,
                )
            )
            data_dict["edge_label_batch"].append(
                torch.full(
                    (d.edge_label.size(0),),
                    idx,
                    dtype=torch.long,
                    device=d.edge_label.device,
                )
            )
            data_dict["atom_label_batch"].append(
                torch.full(
                    (d.atom_label.size(0),),
                    idx,
                    dtype=torch.long,
                    device=d.atom_label.device,
                )
            )
            compose_pos_cusum += d.cpx_pos.size(0)
            edge_query_index_cusum += d.edge_query_index_0.size(0)
            pos_fake_cusum += d.pos_fake.size(0)
            pos_real_cusum += d.pos_real.size(0)

        data_dict["cpx_pos"] = torch.cat(data_dict["cpx_pos"])
        data_dict["cpx_feature"] = torch.cat(data_dict["cpx_feature"])
        data_dict["idx_ligand_ctx_in_cpx"] = torch.cat(data_dict["idx_ligand_ctx_in_cpx"])
        data_dict["idx_protein_in_cpx"] = torch.cat(data_dict["idx_protein_in_cpx"])
        data_dict["candidate_focal_idx_in_protein"] = torch.cat(data_dict["candidate_focal_idx_in_protein"])
        data_dict["candidate_focal_label_in_protein"] = torch.cat(
            data_dict["candidate_focal_label_in_protein"]
        )
        data_dict["focal_idx_in_context_selected"] = torch.cat(data_dict["focal_idx_in_context_selected"])
        data_dict["focal_idx_in_context_candidates"] = torch.cat(data_dict["focal_idx_in_context_candidates"])
        data_dict["cpx_edge_index"] = torch.cat(data_dict["cpx_edge_index"], dim=1)
        data_dict["cpx_edge_type"] = torch.cat(data_dict["cpx_edge_type"])
        data_dict["cpx_edge_feature"] = torch.cat(data_dict["cpx_edge_feature"])
        data_dict["cpx_backbone_index"] = torch.cat(data_dict["cpx_backbone_index"])
        data_dict["focal_label"] = torch.cat(data_dict["focal_label"])
        data_dict["ligand_frontier"] = torch.cat(data_dict["ligand_frontier"])
        data_dict["y_pos"] = torch.cat(data_dict["y_pos"], dim=0)
        data_dict["edge_label"] = torch.cat(data_dict["edge_label"])
        data_dict["atom_label"] = torch.cat(data_dict["atom_label"])
        data_dict["edge_query_index_0"] = torch.cat(data_dict["edge_query_index_0"])
        data_dict["edge_query_index_1"] = torch.cat(data_dict["edge_query_index_1"])
        data_dict["pos_query_knn_edge_idx_0"] = torch.cat(data_dict["pos_query_knn_edge_idx_0"])
        data_dict["pos_query_knn_edge_idx_1"] = torch.cat(data_dict["pos_query_knn_edge_idx_1"])
        data_dict["pos_fake"] = torch.cat(data_dict["pos_fake"], dim=0)
        data_dict["pos_fake_knn_edge_idx_0"] = torch.cat(data_dict["pos_fake_knn_edge_idx_0"])
        data_dict["pos_fake_knn_edge_idx_1"] = torch.cat(data_dict["pos_fake_knn_edge_idx_1"])
        data_dict["pos_real"] = torch.cat(data_dict["pos_real"], dim=0)
        data_dict["pos_real_knn_edge_idx_0"] = torch.cat(data_dict["pos_real_knn_edge_idx_0"])
        data_dict["pos_real_knn_edge_idx_1"] = torch.cat(data_dict["pos_real_knn_edge_idx_1"])
        data_dict["index_real_cps_edge_for_atten"] = torch.cat(
            data_dict["index_real_cps_edge_for_atten"],
            dim=1,
        )
        data_dict["tri_edge_index"] = torch.cat(data_dict["tri_edge_index"], dim=1)
        data_dict["tri_edge_feat"] = torch.cat(data_dict["tri_edge_feat"], dim=0)
        data_dict["apo_protein_idx"] = torch.cat(data_dict["apo_protein_idx"])
        data_dict["step_batch"] = torch.cat(data_dict["step_batch"])
        # batch vectors: keep `step_batch` authoritative; retain `*_batch` for compatibility.
        data_dict["cpx_pos_batch"] = data_dict["step_batch"]
        data_dict["y_pos_batch"] = torch.cat(data_dict["y_pos_batch"])
        data_dict["edge_label_batch"] = torch.cat(data_dict["edge_label_batch"])
        data_dict["atom_label_batch"] = torch.cat(data_dict["atom_label_batch"])
        result = cls.from_dict(data_dict)
        if validate:
            result.validate(strict=True)
        del data_dict
        return result


def make_batch_collate(
    *,
    follow_batch: Sequence[str] = (),
    exclude_keys: Sequence[str] = (),
) -> Callable[[Sequence[ComplexData]], Batch]:
    """
    Build a `collate_fn` that batches `ComplexData` via PyG `Batch`.

    This is a small convenience to adapt `torch.utils.data.DataLoader` to
    `torch_geometric.data.Batch.from_data_list`, while allowing callers to
    configure `follow_batch` and `exclude_keys`.

    Args:
        follow_batch: Field names for which PyG should additionally create
            `*_batch` assignment vectors (see PyG `follow_batch`).
        exclude_keys: Field names to omit from batching (useful for large
            Python-only metadata or debugging fields).

    Returns:
        A callable suitable for `DataLoader(..., collate_fn=...)` that converts a
        sequence of `ComplexData` into a single `Batch`.
    """
    follow_batch_list = list(follow_batch)
    exclude_keys_list = list(exclude_keys)

    def collate(data_list: Sequence[ComplexData]) -> Batch:
        data_list_base: Sequence[BaseData] = data_list
        return Batch.from_data_list(
            list(data_list_base),
            follow_batch=follow_batch_list or None,
            exclude_keys=exclude_keys_list or None,
        )

    return collate


class ProteinLigandDataLoader(DataLoader):
    """
    `DataLoader` preconfigured for `ComplexData` + PyG `Batch` collation.

    The default `follow_batch` tracks `ligand_element` and `protein_element`,
    which causes PyG to additionally produce `ligand_element_batch` and
    `protein_element_batch` vectors on the resulting batch.
    """

    def __init__(
        self,
        dataset: Dataset[ComplexData],
        batch_size: int = 1,
        shuffle: bool = False,
        follow_batch: Sequence[str] = ("ligand_element", "protein_element"),
        exclude_keys: Sequence[str] = (),
        **kwargs: Any,
    ) -> None:
        """
        Create a loader that yields `torch_geometric.data.Batch` objects.

        Args:
            dataset: Dataset yielding `ComplexData` examples.
            batch_size: Number of examples per batch.
            shuffle: Whether to shuffle examples each epoch.
            follow_batch: Passed to `Batch.from_data_list` (see `make_batch_collate`).
            exclude_keys: Passed to `Batch.from_data_list` (see `make_batch_collate`).
            **kwargs: Forwarded to `torch.utils.data.DataLoader`.
        """
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=make_batch_collate(
                follow_batch=follow_batch,
                exclude_keys=exclude_keys,
            ),
            **kwargs,
        )


def batch_from_data_list(data_list: Sequence[BaseData]) -> Batch:
    """
    Batch PyG `Data`/`BaseData` objects using PocketFlow defaults.

    Args:
        data_list: Sequence of PyG `BaseData` objects.

    Returns:
        A `Batch` created via `Batch.from_data_list` using
        `follow_batch=["ligand_element", "protein_element"]`.
    """
    return Batch.from_data_list(list(data_list), follow_batch=["ligand_element", "protein_element"])


def torchify_dict(data: Mapping[str, Any]) -> dict[str, Any]:
    """
    Convert NumPy arrays in a mapping to Torch tensors.

    This is a shallow conversion: only values that are `np.ndarray` are
    converted using `torch.from_numpy`; all other values are passed through.

    Args:
        data: Mapping of keys to values, typically produced by preprocessing or
            file IO.

    Returns:
        A new `dict` with the same keys, where NumPy arrays have been converted
        to `torch.Tensor` (CPU) sharing memory with the original arrays when
        possible.
    """
    output: dict[str, Any] = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            output[k] = torch.from_numpy(v)
        else:
            output[k] = v
    return output
