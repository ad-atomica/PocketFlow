"""
Transforms and featurization utilities for PocketFlow complex graphs.

This module defines a set of PyG-style transforms that prepare `ComplexData`
instances for training and generation. The transforms cover:

- Filtering hydrogen atoms and updating ligand topology.
- Counting neighbors and featurizing protein/ligand atoms.
- Building ligand growth trajectories with masked nodes.
- Selecting focal atoms and constructing edge/triangle labels.
- Composing ligand context and protein atoms into a complex graph.
- Collating trajectory steps into a single `ComplexData` batch.

Most transforms mutate the input `ComplexData` in place and return it (or a list
of derived `ComplexData` objects).
"""

from __future__ import annotations

import copy
import operator
from collections.abc import Callable, Sequence
from enum import StrEnum
from itertools import compress
from typing import Any, overload

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.transforms import BaseTransform

from pocket_flow.utils.data import ComplexData, ComplexDataTrajectory
from pocket_flow.utils.neighbor_search import knn, radius
from pocket_flow.utils.transform_utils import (
    GraphType,
    count_neighbors,
    get_bfs_perm,
    get_complex_graph,
    get_complex_graph_,
    get_rfs_perm,
    get_tri_edges,
    make_pos_label,
    mask_node,
    sample_edge_with_radius,
)


class PermType(StrEnum):
    """Permutation types used to order ligand atoms for trajectory building."""

    RFS = "rfs"
    BFS = "bfs"
    MIX = "mix"


class TrajCompose(BaseTransform):
    """Compose multiple transforms that may return trajectory-style lists."""

    type TrajComposeTransform = (
        Callable[[ComplexData], ComplexData]
        | Callable[[ComplexData], list[ComplexData]]
        | Callable[[Sequence[ComplexData]], list[ComplexData]]
        | Callable[[Sequence[ComplexData]], ComplexDataTrajectory]
    )

    def __init__(
        self,
        transforms: Sequence[TrajComposeTransform],
    ) -> None:
        """
        Create a transform composer.

        Args:
            transforms: Callables applied sequentially to the input. If a
                transform returns a list/tuple, the list is flattened into the
                running output list.
        """
        self.transforms = list(transforms)

    @overload
    def forward(self, data: ComplexData) -> ComplexData | list[ComplexData] | ComplexDataTrajectory: ...

    @overload
    def forward(
        self, data: list[ComplexData] | tuple[ComplexData, ...]
    ) -> list[ComplexData | ComplexDataTrajectory]: ...

    def forward(
        self, data: ComplexData | list[ComplexData] | tuple[ComplexData, ...]
    ) -> ComplexData | list[ComplexData] | ComplexDataTrajectory | list[ComplexData | ComplexDataTrajectory]:
        """
        Apply the composed transforms to a single object or a list/tuple.

        Args:
            data: A `ComplexData` or a list/tuple of `ComplexData`.

        Returns:
            If the input is a single `ComplexData`, returns either a single
            `ComplexData`, a list of `ComplexData` steps, or a
            `ComplexDataTrajectory` depending on the composed pipeline.

            If the input is a list/tuple of `ComplexData`, returns a flattened
            list where any list/tuple outputs from transforms are expanded.
        """

        def _apply(obj: ComplexData) -> Any:
            result: Any = obj
            for transform in self.transforms:
                result = transform(result)
            return result

        if isinstance(data, (list, tuple)):
            results: list[Any] = []
            for item in data:
                out = _apply(item)
                if isinstance(out, (list, tuple)):
                    results.extend(out)
                else:
                    results.append(out)
            return results

        return _apply(data)

    def __repr__(self) -> str:
        args = [f"  {transform}" for transform in self.transforms]
        return "{}([\n{}\n])".format(self.__class__.__name__, ",\n".join(args))


class RefineData:
    """Remove hydrogen atoms from protein/ligand fields and fix topology."""

    def __init__(self) -> None:
        pass

    def __call__(self, data: ComplexData) -> ComplexData:
        """
        Remove hydrogen atoms and update ligand neighbor/bond structures.

        The method filters protein and ligand atoms where the element equals
        1 (hydrogen). For ligands, it also reindexes:

        - `ligand_nbh_list`
        - `ligand_bond_index`
        - `ligand_bond_type`

        Args:
            data: Complex example with protein/ligand fields populated.

        Returns:
            The same `ComplexData` instance with hydrogen atoms removed.
        """
        # delete H atom of pocket
        protein_element = data.protein_element
        is_H_protein = protein_element == 1
        if is_H_protein.any():
            not_H_protein = ~is_H_protein
            mask_list = not_H_protein.cpu().tolist()
            data.protein_atom_name = list(compress(data.protein_atom_name, mask_list))
            data.protein_atom_to_aa_type = data.protein_atom_to_aa_type[not_H_protein]
            data.protein_element = data.protein_element[not_H_protein]
            data.protein_is_backbone = data.protein_is_backbone[not_H_protein]
            data.protein_pos = data.protein_pos[not_H_protein]
        # delete H atom of ligand
        ligand_element = data.ligand_element
        is_H_ligand = ligand_element == 1
        if is_H_ligand.any():
            not_H_ligand = ~is_H_ligand
            data.ligand_atom_feature = data.ligand_atom_feature[not_H_ligand]
            data.ligand_element = data.ligand_element[not_H_ligand]
            data.ligand_pos = data.ligand_pos[not_H_ligand]
            # nbh
            device = data.ligand_element.device
            index_atom_H = torch.nonzero(is_H_ligand, as_tuple=False).view(-1)
            index_atom_H_set = set(index_atom_H.tolist())

            index_changer = torch.full((not_H_ligand.numel(),), -1, dtype=torch.long, device=device)
            index_changer[not_H_ligand] = torch.arange(
                int(not_H_ligand.sum().item()), dtype=torch.long, device=device
            )

            not_H_ligand_cpu = not_H_ligand.cpu().tolist()
            index_changer_cpu = index_changer.cpu().tolist()
            old_nbh = data.ligand_nbh_list
            new_nbh: dict[int, list[int]] = {}
            for old_i, keep in enumerate(not_H_ligand_cpu):
                if not keep:
                    continue
                new_i = index_changer_cpu[old_i]
                neigh = old_nbh.get(old_i, [])
                new_neigh: list[int] = []
                for old_j in neigh:
                    if old_j in index_atom_H_set:
                        continue
                    if not_H_ligand_cpu[old_j]:
                        new_neigh.append(index_changer_cpu[old_j])
                new_nbh[new_i] = new_neigh
            data.ligand_nbh_list = new_nbh
            # bond
            bond_i, bond_j = data.ligand_bond_index
            ind_bond_without_H = ~(is_H_ligand[bond_i] | is_H_ligand[bond_j])
            old_ligand_bond_index = data.ligand_bond_index[:, ind_bond_without_H]
            data.ligand_bond_index = index_changer[old_ligand_bond_index]
            data.ligand_bond_type = data.ligand_bond_type[ind_bond_without_H]
        return data


class LigandCountNeighbors:
    """Compute neighbor counts, valence, and bond-type counts for ligands."""

    def __init__(self) -> None:
        pass

    def __call__(self, data: ComplexData) -> ComplexData:
        """
        Attach neighbor-count features for each ligand atom.

        This populates:

        - `ligand_num_neighbors`: total neighbor count (symmetry assumed).
        - `ligand_atom_valence`: sum of bond orders per atom.
        - `ligand_atom_num_bonds`: counts of single/double/triple bonds.

        Args:
            data: Complex example with ligand bond indices/types.

        Returns:
            The same `ComplexData` with neighbor-related fields set.
        """
        data.ligand_num_neighbors = count_neighbors(
            data.ligand_bond_index,
            symmetry=True,
            num_nodes=data.ligand_element.size(0),
        )
        data.ligand_atom_valence = count_neighbors(
            data.ligand_bond_index,
            symmetry=True,
            valence=data.ligand_bond_type,
            num_nodes=data.ligand_element.size(0),
        )
        data.ligand_atom_num_bonds = torch.stack(
            [
                count_neighbors(
                    data.ligand_bond_index,
                    symmetry=True,
                    valence=(data.ligand_bond_type == i).long(),
                    num_nodes=data.ligand_element.size(0),
                )
                for i in [1, 2, 3]
            ],
            dim=-1,
        )
        return data


class FeaturizeProteinAtom:
    """Generate protein atom features from element, residue, and backbone flags."""

    def __init__(self) -> None:
        self.atomic_numbers = torch.LongTensor([6, 7, 8, 16, 34])  # C, N, O, S, Se
        self.max_num_aa: int = 20

    @property
    def feature_dim(self) -> int:
        """Return the length of the protein atom feature vector."""
        return self.atomic_numbers.size(0) + self.max_num_aa + 1 + 1

    def __call__(self, data: ComplexData) -> ComplexData:
        """
        Build and attach `protein_atom_feature`.

        The feature concatenates:

        - One-hot element encoding for C/N/O/S/Se.
        - One-hot amino acid type (`max_num_aa` classes).
        - Backbone indicator.
        - A trailing `is_mol_atom` flag set to 0 for protein atoms.

        Args:
            data: Complex example with protein element and residue info.

        Returns:
            The same `ComplexData` with `protein_atom_feature` set.
        """
        atomic_numbers = self.atomic_numbers.to(data.protein_element.device)
        element = (
            data.protein_element.view(-1, 1) == atomic_numbers.view(1, -1)
        ).long()  # (N_atoms, N_elements)
        amino_acid = F.one_hot(data.protein_atom_to_aa_type, num_classes=self.max_num_aa)
        is_backbone = data.protein_is_backbone.view(-1, 1).long()
        is_mol_atom = torch.zeros_like(is_backbone, dtype=torch.long)
        x = torch.cat([element, amino_acid, is_backbone, is_mol_atom], dim=-1)
        data.protein_atom_feature = x
        return data


class FeaturizeLigandAtom:
    """Generate ligand atom features from element and bond-derived counts."""

    def __init__(
        self,
        atomic_numbers: Sequence[int] = (1, 6, 7, 8, 9, 15, 16, 17, 35, 53),
    ) -> None:
        """
        Create a ligand featurizer.

        Args:
            atomic_numbers: Allowed atomic numbers to one-hot encode.
        """
        self.atomic_numbers: torch.Tensor = torch.LongTensor(list(atomic_numbers))

    @property
    def feature_dim(self) -> int:
        """Return the length of the ligand atom feature vector."""
        return self.atomic_numbers.size(0) + (1 + 1 + 1) + 3

    def __call__(self, data: ComplexData) -> ComplexData:
        """
        Build and attach `ligand_atom_feature_full`.

        The feature concatenates:

        - One-hot element encoding using `atomic_numbers`.
        - `is_mol_atom` flag set to 1 for ligand atoms.
        - Neighbor count and valence.
        - Counts of single/double/triple bonds.

        Args:
            data: Complex example with ligand element and neighbor statistics.

        Returns:
            The same `ComplexData` with `ligand_atom_feature_full` set.
        """
        atomic_numbers = self.atomic_numbers.to(data.ligand_element.device)
        element = (
            data.ligand_element.view(-1, 1) == atomic_numbers.view(1, -1)
        ).long()  # (N_atoms, N_elements)
        is_mol_atom = torch.ones((element.size(0), 1), dtype=torch.long, device=element.device)
        n_neigh = data.ligand_num_neighbors.view(-1, 1).long()
        n_valence = data.ligand_atom_valence.view(-1, 1).long()
        ligand_atom_num_bonds = data.ligand_atom_num_bonds
        x = torch.cat([element, is_mol_atom, n_neigh, n_valence, ligand_atom_num_bonds], dim=-1)
        data.ligand_atom_feature_full = x
        return data


class LigandTrajectory:
    """Generate a sequence of masked ligand-growth steps from a permutation."""

    def __init__(
        self,
        perm_type: PermType = PermType.BFS,
        p: list[float] | None = None,
        num_atom_type: int = 10,
        y_pos_std: float = 0.05,
    ) -> None:
        """
        Configure ligand trajectory generation.

        Args:
            perm_type: Strategy for atom ordering (RFS, BFS, or MIX).
            p: Sampling probabilities for MIX mode, ordered as [RFS, BFS].
            num_atom_type: Number of element types to encode in masking.
            y_pos_std: Position noise standard deviation for masked atoms.
        """
        self.perm_type = perm_type
        self.num_atom_type = num_atom_type
        self.y_pos_std = y_pos_std
        self.p = p

    def __call__(self, data: ComplexData) -> list[ComplexData]:
        """
        Build a step-by-step ligand masking trajectory.

        Each step deep-copies the input `data` and masks a prefix of the
        permutation. The returned list length matches the number of ligand
        atoms in the permutation.

        Args:
            data: Complex example with ligand neighbor list and ring info.

        Returns:
            A list of `ComplexData` objects, one per growth step.
        """
        if self.perm_type == PermType.RFS:
            perm, _edge_index = get_rfs_perm(data.ligand_nbh_list, data.ligand_ring_info)
        elif self.perm_type == PermType.BFS:
            perm, _edge_index = get_bfs_perm(data.ligand_nbh_list)
        elif self.perm_type == PermType.MIX:
            perm_type_choice = np.random.choice([PermType.RFS, PermType.BFS], p=self.p)
            if perm_type_choice == PermType.RFS:
                perm, _edge_index = get_rfs_perm(data.ligand_nbh_list, data.ligand_ring_info)
            else:
                perm, _edge_index = get_bfs_perm(data.ligand_nbh_list)
        traj = []
        for ix, _i in enumerate(perm):
            data_step = copy.deepcopy(data)
            if ix == 0:
                out = mask_node(
                    data_step,
                    torch.empty([0], dtype=torch.long),
                    perm,
                    num_atom_type=self.num_atom_type,
                    y_pos_std=self.y_pos_std,
                )
                traj.append(out)
            else:
                out = mask_node(
                    data_step,
                    perm[:ix],
                    perm[ix:],
                    num_atom_type=self.num_atom_type,
                    y_pos_std=self.y_pos_std,
                )
                traj.append(out)
        del data
        return traj


class FocalMaker:
    """Select focal atoms and build local edge/triangle labels for each step."""

    def __init__(
        self,
        r: float = 4.0,
        num_work: int = 16,
        atomic_numbers: Sequence[int] = (1, 6, 7, 8, 9, 15, 16, 17, 35, 53),
    ) -> None:
        """
        Create a focal maker for ligand-growth steps.

        Args:
            r: Radius (in Angstroms) for candidate focal atom search in protein.
            num_work: Worker count for neighbor search operations.
            atomic_numbers: Atomic numbers to map to atom labels.
        """
        self.r: float = r
        self.num_work: int = num_work
        self.atomic_numbers: torch.Tensor = torch.LongTensor(list(atomic_numbers))

    def run(self, data: ComplexData) -> ComplexData:
        """
        Populate focal-related labels for a single trajectory step.

        The method handles two cases:

        - If no ligand context exists yet, it selects a protein atom closest to
          the masked ligand position and builds candidate focal labels.
        - Otherwise, it selects a focal atom from ligand-context neighbors and
          creates edge labels, triangle edges, and candidate focal sets.

        Args:
            data: A single-step `ComplexData` produced by masking.

        Returns:
            The same `ComplexData` with focal and edge label fields set.
        """
        if not hasattr(data, "idx_ligand_ctx_in_cpx"):
            data.idx_ligand_ctx_in_cpx = torch.arange(
                data.ligand_context_pos.size(0), dtype=torch.long, device=data.ligand_context_pos.device
            )
        if data.ligand_context_pos.size(0) == 0:
            device = data.protein_pos.device
            masked_pos = data.ligand_pos[data.masked_idx[0]]
            focal_idx_in_context_selected = torch.norm(
                data.protein_pos - masked_pos.unsqueeze(0), p=2, dim=-1
            ).argmin()
            data.focal_idx_in_context_selected = focal_idx_in_context_selected.unsqueeze(0)
            data.focal_idx_in_context_candidates = focal_idx_in_context_selected.unsqueeze(0)
            element_match = data.ligand_element[data.masked_idx[0]] == self.atomic_numbers
            data.atom_label = torch.nonzero(element_match, as_tuple=False).view(-1)
            data.edge_label = torch.empty(0, dtype=torch.long, device=device)
            data.focal_label = torch.empty(0, dtype=torch.long, device=device)
            data.edge_query_index_0 = torch.zeros_like(data.edge_label)
            data.edge_query_index_1 = torch.arange(data.edge_label.size(0), device=device)
            edge_index_query = torch.stack([data.edge_query_index_0, data.edge_query_index_1])
            data.index_real_cps_edge_for_atten, data.tri_edge_index, data.tri_edge_feat = get_tri_edges(
                edge_index_query,
                data.y_pos,
                data.idx_ligand_ctx_in_cpx,
                data.ligand_context_bond_index,
                data.ligand_context_bond_type,
            )
            # candidate focal atom without ligand
            assign_index = radius(
                x=data.ligand_masked_pos, y=data.protein_pos, r=self.r, num_workers=self.num_work
            )
            if assign_index.size(1) == 0:
                dist = torch.norm(
                    data.protein_pos.unsqueeze(1) - data.ligand_masked_pos.unsqueeze(0), p=2, dim=-1
                )
                assign_index = torch.nonzero(dist <= torch.min(dist) + 1e-5)[0:1].transpose(0, 1)
            data.candidate_focal_idx_in_protein = torch.unique(assign_index[0])
            candidate_focal_label_in_protein = torch.zeros(
                data.protein_pos.size(0), dtype=torch.bool, device=device
            )
            candidate_focal_label_in_protein[data.candidate_focal_idx_in_protein] = True
            data.candidate_focal_label_in_protein = candidate_focal_label_in_protein
            data.apo_protein_idx = torch.arange(data.protein_pos.size(0), dtype=torch.long, device=device)
        else:
            new_step_atom_idx = data.masked_idx[0]
            device = data.context_idx.device
            candidate_raw = data.ligand_nbh_list.get(operator.index(new_step_atom_idx), [])
            candidate_focal_idx_in_context = torch.as_tensor(candidate_raw, dtype=torch.long, device=device)
            if candidate_focal_idx_in_context.numel() > 0:
                focal_idx_in_context_candidates_mask = (
                    data.context_idx.unsqueeze(1) == candidate_focal_idx_in_context
                ).any(1)
                data.focal_idx_in_context_candidates = torch.nonzero(
                    focal_idx_in_context_candidates_mask, as_tuple=False
                ).view(-1)
            else:
                data.focal_idx_in_context_candidates = torch.empty(0, dtype=torch.long, device=device)
            if data.focal_idx_in_context_candidates.numel() == 0:
                masked_pos = data.ligand_pos[new_step_atom_idx].view(1, 3)
                dist = torch.norm(data.ligand_context_pos - masked_pos, p=2, dim=-1)
                data.focal_idx_in_context_candidates = dist.argmin().view(-1)
            weights = torch.ones(
                (data.focal_idx_in_context_candidates.size(0),), dtype=torch.float, device=device
            )
            focal_choice_idx = torch.multinomial(weights, 1)
            data.focal_idx_in_context_selected = data.focal_idx_in_context_candidates[focal_choice_idx]
            data.focal_label = torch.zeros_like(data.context_idx)
            data.focal_label[data.focal_idx_in_context_selected] = 1
            element_match = data.ligand_element[data.masked_idx[0]] == self.atomic_numbers
            data.atom_label = torch.nonzero(element_match, as_tuple=False).view(-1)
            data.apo_protein_idx = torch.empty(0, dtype=torch.long, device=device)
            data.candidate_focal_idx_in_protein = torch.empty(0, dtype=torch.long, device=device)
            data.candidate_focal_label_in_protein = torch.empty(0, dtype=torch.bool, device=device)
            # get edge label
            data = sample_edge_with_radius(data, r=4.0, num_workers=self.num_work)
            # get triangle edge
            if data.idx_ligand_ctx_in_cpx.numel() == 0:
                edge_index_query_ctx = torch.stack([data.edge_query_index_0, data.edge_query_index_1])
            else:
                max_cpx = int(data.idx_ligand_ctx_in_cpx.max().item())
                cpx_to_ctx = torch.full((max_cpx + 1,), -1, dtype=torch.long, device=device)
                cpx_to_ctx[data.idx_ligand_ctx_in_cpx] = torch.arange(
                    data.idx_ligand_ctx_in_cpx.numel(), device=device
                )
                edge_query_ctx = cpx_to_ctx[data.edge_query_index_1]
                assert (edge_query_ctx >= 0).all().item()
                edge_index_query_ctx = torch.stack([data.edge_query_index_0, edge_query_ctx])
            data.index_real_cps_edge_for_atten, data.tri_edge_index, data.tri_edge_feat = get_tri_edges(
                edge_index_query_ctx,
                data.y_pos,
                data.idx_ligand_ctx_in_cpx,
                data.ligand_context_bond_index,
                data.ligand_context_bond_type,
            )
        return data

    def __call__(self, data_list: list[ComplexData]) -> list[ComplexData]:
        """
        Apply `run` to each element of a trajectory list.

        Args:
            data_list: List of single-step `ComplexData` objects.

        Returns:
            A new list with focal labels added to each step.
        """
        data_list_new = []
        for i in data_list:
            data_list_new.append(self.run(i))
        del data_list
        return data_list_new


class AtomComposer:
    """Compose ligand-context and protein atoms into a complex graph."""

    def __init__(
        self,
        knn: int = 16,
        *,
        num_workers: int,
        graph_type: GraphType = GraphType.KNN,
        radius: float = 10.0,
        num_real_pos: int = 5,
        num_fake_pos: int = 5,
        pos_real_std: float = 0.05,
        pos_fake_std: float = 2.0,
        for_gen: bool = False,
        use_protein_bond: bool = True,
    ) -> None:
        """
        Configure complex-graph composition.

        Args:
            knn: Number of neighbors for KNN-based graph construction.
            num_workers: Worker count for neighbor search operations.
            graph_type: Graph construction strategy (KNN or radius-based).
            radius: Radius for radius-based graph construction.
            num_real_pos: Number of positive position samples per step.
            num_fake_pos: Number of negative position samples per step.
            pos_real_std: Noise for real position samples.
            pos_fake_std: Noise for fake position samples.
            for_gen: If True, skip position label sampling (generation mode).
            use_protein_bond: Whether to include protein bonds in the graph.
        """
        self.graph_type: GraphType = graph_type
        self.radius: float = radius
        self.knn: int = knn
        self.num_workers: int = num_workers
        self.num_real_pos: int = num_real_pos
        self.num_fake_pos: int = num_fake_pos
        self.pos_real_std: float = pos_real_std
        self.pos_fake_std: float = pos_fake_std
        self.for_gen: bool = for_gen
        self.use_protein_bond: bool = use_protein_bond

    def run(self, data: ComplexData) -> ComplexData:
        """
        Build complex graph tensors for a single trajectory step.

        The method concatenates ligand-context and protein features/positions,
        builds the complex graph edges, and optionally creates position labels
        for training.

        Args:
            data: A single-step `ComplexData` with context features populated.

        Returns:
            The same `ComplexData` with complex graph and label fields set.
        """
        protein_feat_dim = data.protein_atom_feature.size(-1)
        ligand_feat_dim = data.ligand_context_feature_full.size(-1)
        num_ligand_ctx_atom = data.ligand_context_feature_full.size(0)
        num_protein_atom = data.protein_atom_feature.size(0)
        device = data.ligand_context_feature_full.device

        data.cpx_pos = torch.cat([data.ligand_context_pos, data.protein_pos], dim=0)
        data.step_batch = torch.zeros(data.cpx_pos.size(0), dtype=torch.long, device=data.cpx_pos.device)
        num_complex_atom = data.cpx_pos.size(0)
        feat_dim = max(protein_feat_dim, ligand_feat_dim)

        ligand_feat = data.ligand_context_feature_full
        if ligand_feat_dim < feat_dim:
            pad = torch.zeros(
                (num_ligand_ctx_atom, feat_dim - ligand_feat_dim),
                dtype=ligand_feat.dtype,
                device=ligand_feat.device,
            )
            ligand_feat = torch.cat([ligand_feat, pad], dim=1)

        protein_feat = data.protein_atom_feature
        if protein_feat_dim < feat_dim:
            pad = torch.zeros(
                (num_protein_atom, feat_dim - protein_feat_dim),
                dtype=protein_feat.dtype,
                device=protein_feat.device,
            )
            protein_feat = torch.cat([protein_feat, pad], dim=1)

        data.cpx_feature = torch.cat([ligand_feat, protein_feat], dim=0)
        data.idx_ligand_ctx_in_cpx = torch.arange(num_ligand_ctx_atom, dtype=torch.long, device=device)
        data.idx_protein_in_cpx = (
            torch.arange(num_protein_atom, dtype=torch.long, device=device) + num_ligand_ctx_atom
        )
        assert data.idx_ligand_ctx_in_cpx.numel() == data.ligand_context_pos.size(0)
        if data.idx_ligand_ctx_in_cpx.numel() > 0:
            assert int(data.idx_ligand_ctx_in_cpx.min().item()) >= 0
            assert int(data.idx_ligand_ctx_in_cpx.max().item()) < data.cpx_pos.size(0)
            assert data.idx_ligand_ctx_in_cpx.unique().numel() == data.idx_ligand_ctx_in_cpx.numel()
        if self.use_protein_bond:
            data = get_complex_graph_(
                data,
                knn=self.knn,
                num_workers=self.num_workers,
                graph_type=self.graph_type,
                radius=self.radius,
            )
        else:
            data = get_complex_graph(
                data,
                num_ligand_ctx_atom,
                num_complex_atom,
                num_workers=self.num_workers,
                graph_type=self.graph_type,
                knn=self.knn,
                radius=self.radius,
            )
        pos_query_knn_edge_idx = knn(x=data.cpx_pos, y=data.y_pos, k=self.knn, num_workers=self.num_workers)
        data.pos_query_knn_edge_idx_0, data.pos_query_knn_edge_idx_1 = (
            pos_query_knn_edge_idx[1],
            pos_query_knn_edge_idx[0],
        )
        data.cpx_backbone_index = torch.nonzero(data.protein_is_backbone).view(
            -1
        ) + data.ligand_context_feature_full.size(0)
        if self.for_gen is False:
            data = make_pos_label(
                data,
                num_real_pos=self.num_real_pos,
                num_fake_pos=self.num_fake_pos,
                pos_real_std=self.pos_real_std,
                pos_fake_std=self.pos_fake_std,
                k=self.knn,
                num_workers=self.num_workers,
            )
        return data

    def __call__(self, data_list: list[ComplexData]) -> list[ComplexData]:
        """
        Apply `run` to each element of a trajectory list.

        Args:
            data_list: List of single-step `ComplexData` objects.

        Returns:
            A new list with complex graph fields added to each step.
        """
        d_list_new = []
        for d in data_list:
            d_list_new.append(self.run(d))
        del data_list
        return d_list_new


class Combine:
    """Convenience transform that runs trajectory, focal, and compose steps."""

    def __init__(
        self,
        lig_traj: LigandTrajectory,
        focal_maker: FocalMaker,
        atom_composer: AtomComposer,
        lig_only: bool = False,
    ) -> None:
        """
        Create a combined transform.

        Args:
            lig_traj: Trajectory generator for ligand masking.
            focal_maker: Focal atom selection and edge labeling helper.
            atom_composer: Complex graph composition helper.
            lig_only: If True, skip the initial empty-context step.
        """
        self.lig_traj: LigandTrajectory = lig_traj
        self.focal_maker: FocalMaker = focal_maker
        self.atom_composer: AtomComposer = atom_composer
        self.lig_only: bool = lig_only

    def __call__(self, data: ComplexData) -> list[ComplexData]:
        """
        Generate a trajectory and augment it with focal and graph features.

        Args:
            data: Complex example with ligand and protein information.

        Returns:
            A list of `ComplexData` steps after masking, focal selection, and
            complex graph composition.
        """
        if self.lig_traj.perm_type == PermType.RFS:
            perm, _edge_index = get_rfs_perm(data.ligand_nbh_list, data.ligand_ring_info)
        elif self.lig_traj.perm_type == PermType.BFS:
            perm, _edge_index = get_bfs_perm(data.ligand_nbh_list)
        elif self.lig_traj.perm_type == PermType.MIX:
            perm_type_choice = np.random.choice([PermType.RFS, PermType.BFS], p=self.lig_traj.p)
            if perm_type_choice == PermType.RFS:
                perm, _edge_index = get_rfs_perm(data.ligand_nbh_list, data.ligand_ring_info)
            else:
                perm, _edge_index = get_bfs_perm(data.ligand_nbh_list)
        traj = []
        for ix, _i in enumerate(perm):
            data_step = copy.deepcopy(data)
            if ix == 0:
                if not self.lig_only:
                    out = mask_node(
                        data_step,
                        torch.empty([0], dtype=torch.long),
                        perm,
                        num_atom_type=self.lig_traj.num_atom_type,
                        y_pos_std=self.lig_traj.y_pos_std,
                    )
                else:
                    continue
            else:
                out = mask_node(
                    data_step,
                    perm[:ix],
                    perm[ix:],
                    num_atom_type=self.lig_traj.num_atom_type,
                    y_pos_std=self.lig_traj.y_pos_std,
                )
            out = self.focal_maker.run(out)
            out = self.atom_composer.run(out)
            traj.append(out)
        del data
        return traj
