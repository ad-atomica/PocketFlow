"""
Graph/geometry helpers used by PocketFlow data transforms.

This module provides a small set of utilities that are used by
`pocket_flow/utils/transform.py` to build step-wise ligand trajectories for
autoregressive generation and to construct the composed ligand+protein context
graph consumed by the model.

Most helpers operate on a PyG :class:`torch_geometric.data.Data` object and
*mutate it in-place* by adding derived attributes (e.g., context/masked subsets,
edge labels, and auxiliary kNN/radius edges). Shape conventions follow PyG:

- Edge indices are stored as ``(2, E)`` tensors, where the first row contains
  source indices and the second row contains destination indices.
- For "query → context" edges produced by :func:`torch_geometric.nn.knn` and
  :func:`torch_geometric.nn.radius`, the returned index tensor uses
  ``edge_index[0]`` for indices into the query set (``y``) and ``edge_index[1]``
  for indices into the reference set (``x``).
  When these edges are used for message passing from the reference set to the
  query set, this code swaps the returned rows to maintain the source→target
  ordering.

Randomness:
    Several functions sample random starting nodes or random positions (NumPy
    and Python's ``random`` module are used). For reproducibility, ensure all
    relevant RNGs are seeded by the caller.
"""

from __future__ import annotations

import copy
import random
from enum import StrEnum
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import scatter as pyg_scatter
from torch_geometric.utils import subgraph
from torch_geometric.utils.num_nodes import maybe_num_nodes

from pocket_flow.utils.data import ComplexData, LigandRingInfo, NeighborList
from pocket_flow.utils.neighbor_search import knn, knn_graph, radius, radius_graph

if TYPE_CHECKING:
    from numpy.typing import NDArray


class GraphType(StrEnum):
    """Graph construction type for kNN or radius-based graphs."""

    KNN = "knn"
    RAD = "rad"


def count_neighbors(
    edge_index: torch.Tensor,
    symmetry: bool,
    valence: torch.Tensor | None = None,
    num_nodes: int | None = None,
) -> torch.Tensor:
    """Count per-node neighbors (or weighted neighbor sums) from an edge list.

    This is a lightweight wrapper around :func:`torch_geometric.utils.scatter`.
    The implementation assumes *symmetrical* (undirected) edges where both
    directions are explicitly present in ``edge_index``.

    Args:
        edge_index: Edge index tensor of shape ``(2, E)``.
        symmetry: Must be `True`. This function currently only supports graphs
            where each undirected edge is represented twice (``i→j`` and
            ``j→i``).
        valence: Optional per-edge weights of shape ``(E,)`` (or broadcastable
            to that shape). When omitted, each edge contributes weight 1.
        num_nodes: Number of nodes ``N``. When omitted, it is inferred via
            :func:`torch_geometric.utils.num_nodes.maybe_num_nodes`.

    Returns:
        A ``(N,)`` integer tensor where entry ``i`` is the sum of weights over
        edges whose source is ``i``.
    """
    assert symmetry, "Only support symmetrical edges."

    if num_nodes is None:
        num_nodes = maybe_num_nodes(edge_index)

    if valence is None:
        valence = torch.ones([edge_index.size(1)], device=edge_index.device)
    valence = valence.view(edge_index.size(1))
    return pyg_scatter(valence, edge_index[0], dim=0, dim_size=num_nodes, reduce="sum").long()


def change_features_of_neigh(
    ligand_feature_full: torch.Tensor,
    new_num_neigh: torch.Tensor,
    new_num_valence: torch.Tensor,
    ligand_atom_num_bonds: torch.Tensor,
    num_atom_type: int = 7,
) -> torch.Tensor:
    """Update neighbor/valence-related columns in a ligand feature matrix.

    The "full" ligand atom feature layout used by this project is:

    - ``[0 : num_atom_type]``: atom-type one-hot (elements)
    - ``[num_atom_type]``: `is_mol_atom` flag (typically 1 for ligand atoms)
    - ``[num_atom_type + 1]``: number of neighbors
    - ``[num_atom_type + 2]``: valence (sum of bond orders/types)
    - ``[num_atom_type + 3 : num_atom_type + 6]``: counts of single/double/triple bonds

    Note:
        The exact indices depend on the ``num_atom_type`` parameter, which should
        match the number of element types used by :class:`FeaturizeLigandAtom`.

    This function overwrites only the neighbor/valence/bond-count columns and
    returns the same tensor object for convenience.

    Args:
        ligand_feature_full: Feature tensor of shape ``(N, F)``.
        new_num_neigh: Updated neighbor counts, shape ``(N,)``.
        new_num_valence: Updated valence values, shape ``(N,)``.
        ligand_atom_num_bonds: Updated bond-type counts, shape ``(N, 3)``.
        num_atom_type: Number of element one-hot channels.

    Returns:
        The input ``ligand_feature_full`` tensor (mutated in-place).
    """
    idx_n_neigh = num_atom_type + 1
    idx_n_valence = idx_n_neigh + 1
    idx_n_bonds = idx_n_valence + 1
    ligand_feature_full[:, idx_n_neigh] = new_num_neigh.long()
    ligand_feature_full[:, idx_n_valence] = new_num_valence.long()
    ligand_feature_full[:, idx_n_bonds : idx_n_bonds + 3] = ligand_atom_num_bonds.long()
    return ligand_feature_full


def get_rfs_perm(
    nbh_list: NeighborList,
    ring_info: LigandRingInfo,
) -> tuple[torch.Tensor, list[list[tuple[int, int]]]]:
    """Generate a "ring-first" traversal permutation over a molecular graph.

    The traversal starts from a random node. Nodes that participate in rings
    (as indicated by ``ring_info``) are preferentially visited before non-ring
    nodes, with an additional preference to stay within the same ring when
    expanding from a ring node.

    The second return value contains per-step edge lists connecting the newly
    visited node to *previously visited* neighbors (both directions are emitted
    for each such edge). This can be used to construct step-wise subgraphs.

    Args:
        nbh_list: Adjacency list mapping node index → list of neighbor indices.
            Node indices are assumed to be ``0..N-1``.
        ring_info: Ring membership indicator tensor, indexed by node.
            This code assumes a representation where ``(ring_info[i] > 0)``
            indicates that node ``i`` participates in at least one ring, and
            where two nodes share a ring if ``(ring_info[i] == ring_info[j])``
            has any positive entry.

    Returns:
        A tuple ``(perm, step_edges)`` where:
          - ``perm`` is a ``(N,)`` tensor of node indices in visit order.
          - ``step_edges`` is a list of length ``N``; entry ``t`` is a list of
            directed edge tuples ``(u, v)`` created when visiting ``perm[t]``.
    """
    num_nodes = len(nbh_list)
    node0 = random.randint(0, num_nodes - 1)
    queue: list[int] = []
    order: list[int] = []
    not_ring_queue: list[int] = []
    edge_index: list[list[tuple[int, int]]] = []
    if (ring_info[node0] > 0).sum():
        queue.append(node0)
    else:
        not_ring_queue.append(node0)
    while queue or not_ring_queue:
        if queue:
            v = queue.pop()
            order.append(v)
        elif not_ring_queue:
            v = not_ring_queue.pop()
            order.append(v)
        adj_in_ring: list[int] = []
        adj_not_ring: list[int] = []
        edge_idx_step: list[tuple[int, int]] = []
        for nbh in nbh_list[v]:
            if (ring_info[nbh] > 0).sum():
                adj_in_ring.append(nbh)
            else:
                adj_not_ring.append(nbh)
            if nbh in order:
                edge_idx_step.append((v, nbh))
                edge_idx_step.append((nbh, v))
        edge_index.append(edge_idx_step)
        if adj_not_ring:
            for w in adj_not_ring:
                if w not in order and w not in not_ring_queue:
                    not_ring_queue.append(w)
        # Preferential access to atoms in the same ring of w
        same_ring_pool: list[int] = []
        if adj_in_ring:
            for w in adj_in_ring:
                if w not in order and w not in queue:
                    if (ring_info[w] == ring_info[v]).sum() > 0:
                        same_ring_pool.append(w)
                    else:
                        queue.append(w)
                elif w not in order and w in queue:
                    if (ring_info[w] == ring_info[v]).sum() > 0:
                        same_ring_pool.append(w)
                        queue.remove(w)
                elif w not in order and (ring_info[w] > 0).sum() >= 1:
                    queue.remove(w)
                    queue.append(w)
            queue += same_ring_pool
    return torch.LongTensor(order), edge_index


def get_bfs_perm(
    nbh_list: NeighborList,
) -> tuple[torch.Tensor, list[list[tuple[int, int]]]]:
    """Generate a randomized BFS permutation over a molecular adjacency list.

    The BFS root is chosen uniformly at random. Within each BFS layer, the next
    frontier is shuffled before being enqueued, producing a randomized BFS
    order.

    As with :func:`get_rfs_perm`, this returns per-step edge tuples connecting
    the newly visited node to already-visited nodes.

    Args:
        nbh_list: Adjacency list mapping node index → list of neighbor indices.
            Node indices are assumed to be ``0..N-1``.

    Returns:
        A tuple ``(perm, step_edges)`` where:
          - ``perm`` is a ``(N,)`` tensor of node indices in BFS order.
          - ``step_edges`` is a list of length ``N``; entry ``t`` is a list of
            directed edge tuples ``(u, v)`` created when visiting ``perm[t]``.
    """
    num_nodes = len(nbh_list)
    num_neighbors = torch.LongTensor([len(nbh_list[i]) for i in range(num_nodes)])
    bfs_queue = [random.randint(0, num_nodes - 1)]
    bfs_perm: list[int] = []
    edge_index: list[list[tuple[int, int]]] = []
    num_remains = [num_neighbors.clone()]
    bfs_next_list: dict[int, list[int]] = {}
    visited = {bfs_queue[0]}
    num_nbh_remain = num_neighbors.clone()
    while len(bfs_queue) > 0:
        current = bfs_queue.pop(0)
        for nbh in nbh_list[current]:
            num_nbh_remain[nbh] -= 1
        bfs_perm.append(current)
        num_remains.append(num_nbh_remain.clone())
        next_candid: list[int] = []
        edge_idx_step: list[tuple[int, int]] = []
        for nxt in nbh_list[current]:
            if nxt in visited:
                continue
            next_candid.append(nxt)
            visited.add(nxt)
            for adj in nbh_list[nxt]:
                if adj in bfs_perm:
                    edge_idx_step.append((adj, nxt))
                    edge_idx_step.append((nxt, adj))
        edge_index.append(edge_idx_step)
        random.shuffle(next_candid)
        bfs_queue += next_candid
        bfs_next_list[current] = copy.copy(bfs_queue)
    return torch.LongTensor(bfs_perm), edge_index


def mask_node(
    data: ComplexData,
    context_idx: torch.Tensor,
    masked_idx: torch.Tensor,
    num_atom_type: int = 10,
    y_pos_std: float = 0.05,
) -> ComplexData:
    """Split ligand atoms into context vs. masked sets for one generation step.

    This helper is used to build an autoregressive training trajectory. It
    creates context/masked subsets of ligand tensors, re-labels ligand bonds to
    the context subgraph, recomputes context neighbor/valence features, and
    constructs the position target ``data.y_pos`` for the next atom.

    Side effects:
        Mutates and returns the same :class:`~torch_geometric.data.Data` object
        by adding (at least) the attributes listed below.

    Required input attributes on ``data``:
        - ``ligand_element``: ``(N_lig,)`` atomic numbers.
        - ``ligand_atom_feature_full``: ``(N_lig, F)`` full ligand features.
        - ``ligand_pos``: ``(N_lig, 3)`` coordinates.
        - ``ligand_bond_index``: ``(2, E_lig)`` bidirectional bond edges.
        - ``ligand_bond_type``: ``(E_lig,)`` bond types (typically 1/2/3).
        - ``ligand_num_neighbors``: ``(N_lig,)`` neighbor counts in the full ligand.

    Added/updated attributes:
        - ``context_idx`` / ``masked_idx``: input indices saved for later steps.
        - ``ligand_masked_element`` / ``ligand_masked_pos``: masked atom targets.
        - ``ligand_context_element`` / ``ligand_context_pos``: context atoms.
        - ``ligand_context_feature_full``: context features with updated
          neighbor/valence-related columns.
        - ``ligand_context_bond_index`` / ``ligand_context_bond_type``: bonds
          induced by ``context_idx`` (relabelled to ``0..N_ctx-1``).
        - ``ligand_context_num_neighbors`` / ``ligand_context_valence`` /
          ``ligand_context_num_bonds``: recomputed context graph statistics.
        - ``y_pos``: position target for the next atom, shape ``(0, 3)`` when
          there is no masked atom; otherwise ``(1, 3)`` with Gaussian noise.
        - ``ligand_frontier``: boolean mask over context nodes indicating which
          context atoms have at least one missing neighbor relative to the full ligand.

    Args:
        data: PyG data object for a protein–ligand complex.
        context_idx: Indices (in full ligand atom space) that are currently in
            the ligand context.
        masked_idx: Indices (in full ligand atom space) that are masked and to
            be generated later; ``masked_idx[0]`` is treated as "the next atom".
        num_atom_type: Number of element one-hot channels in
            ``ligand_atom_feature_full`` (see :func:`change_features_of_neigh`).
        y_pos_std: Standard deviation of the Gaussian noise added to ``y_pos``.

    Returns:
        The mutated ``data`` object.
    """
    data.context_idx = context_idx  # for change bond index
    data.masked_idx = masked_idx
    # masked ligand atom element/feature/pos.
    data.ligand_masked_element = data.ligand_element[masked_idx]
    # For Prediction. these features are chemical properties
    data.ligand_masked_pos = data.ligand_pos[masked_idx]

    # context ligand atom elment/full features/pos. Note: num_neigh and num_valence features should be changed
    data.ligand_context_element = data.ligand_element[context_idx]
    data.ligand_context_feature_full = data.ligand_atom_feature_full[context_idx]  # For Input
    data.ligand_context_pos = data.ligand_pos[context_idx]

    # new bond with ligand context atoms
    if data.ligand_bond_index.size(1) != 0:
        data.ligand_context_bond_index, data.ligand_context_bond_type = subgraph(
            context_idx,
            data.ligand_bond_index,
            edge_attr=data.ligand_bond_type,
            relabel_nodes=True,
        )
    else:
        data.ligand_context_bond_index = torch.empty([2, 0], dtype=torch.long)
        data.ligand_context_bond_type = torch.empty([0], dtype=torch.long)
    # re-calculate atom features that relate to bond
    data.ligand_context_num_neighbors = count_neighbors(
        data.ligand_context_bond_index,
        symmetry=True,
        num_nodes=context_idx.size(0),
    )
    data.ligand_context_valence = count_neighbors(
        data.ligand_context_bond_index,
        symmetry=True,
        valence=data.ligand_context_bond_type,
        num_nodes=context_idx.size(0),
    )
    data.ligand_context_num_bonds = torch.stack(
        [
            count_neighbors(
                data.ligand_context_bond_index,
                symmetry=True,
                valence=data.ligand_context_bond_type == i,
                num_nodes=context_idx.size(0),
            )
            for i in [1, 2, 3]
        ],
        dim=-1,
    )
    # re-calculate ligand_context_featrure_full
    data.ligand_context_feature_full = change_features_of_neigh(
        data.ligand_context_feature_full,
        data.ligand_context_num_neighbors,
        data.ligand_context_valence,
        data.ligand_context_num_bonds,
        num_atom_type=num_atom_type,
    )
    if data.ligand_masked_pos.size(0) == 0:
        data.y_pos = torch.empty([0, 3], dtype=torch.float32)
    else:
        data.y_pos = data.ligand_masked_pos[0].view(-1, 3)
        data.y_pos += torch.randn_like(data.y_pos) * y_pos_std
    data.ligand_frontier = data.ligand_context_num_neighbors < data.ligand_num_neighbors[context_idx]
    return data


def make_pos_label(
    data: ComplexData,
    num_real_pos: int = 5,
    num_fake_pos: int = 5,
    pos_real_std: float = 0.05,
    pos_fake_std: float = 2.0,
    k: int = 16,
    *,
    num_workers: int,
) -> ComplexData:
    """Sample real/fake query positions and build kNN edges to the composed graph.

    This is used for position scoring objectives: ``pos_real`` are sampled near
    true masked ligand atom positions, and ``pos_fake`` are sampled around
    either the ligand frontier or (when there is no ligand context) around
    candidate protein focal sites.

    Required input attributes on ``data``:
        - ``cpx_pos``: composed positions, shape ``(N_cpx, 3)``.
        - ``protein_pos``: protein coordinates, shape ``(N_prot, 3)``.
        - ``ligand_context_pos``: context ligand coordinates, shape ``(N_ctx, 3)``.
        - ``ligand_masked_pos``: masked ligand coordinates, shape ``(N_mask, 3)``.
        - ``ligand_frontier``: boolean mask over context nodes (when ``N_ctx>0``).
        - ``candidate_focal_label_in_protein``: boolean mask over protein nodes
          used when ``N_ctx==0``.
        - ``y_pos``: target next position, used to ensure one real sample
          matches the current step when ``N_ctx>0``.

    Added attributes:
        - ``pos_fake`` / ``pos_real``: sampled positions, shapes
          ``(num_fake_pos, 3)`` and ``(num_real_pos, 3)``.
        - ``pos_fake_knn_edge_idx_0`` / ``pos_fake_knn_edge_idx_1``:
          kNN edges from ``cpx_pos`` (source) to ``pos_fake`` (target); the
          first index vector refers to ``cpx_pos`` rows and the second refers to
          ``pos_fake`` rows.
        - ``pos_real_knn_edge_idx_0`` / ``pos_real_knn_edge_idx_1``:
          analogous edges from ``cpx_pos`` to ``pos_real``.

    Args:
        data: PyG data object to mutate.
        num_real_pos: Number of real samples to draw.
        num_fake_pos: Number of fake samples to draw.
        pos_real_std: Noise scale for real samples.
        pos_fake_std: Noise scale for fake samples.
        k: Number of nearest neighbors in kNN edges.
        num_workers: Worker count for PyG neighbor search.

    Returns:
        The mutated ``data`` object.
    """
    ligand_context_pos: torch.Tensor = data.ligand_context_pos
    ligand_masked_pos: torch.Tensor = data.ligand_masked_pos
    protein_pos: torch.Tensor = data.protein_pos

    if ligand_context_pos.size(0) == 0:
        # fake position
        fake_mode: torch.Tensor = protein_pos[data.candidate_focal_label_in_protein]
        n_fake = fake_mode.size(0)
        p = np.ones(n_fake, dtype=np.float32) / n_fake
        pos_fake_idx: NDArray[np.intp] = np.random.choice(np.arange(n_fake), size=num_fake_pos, p=p)
        pos_fake: torch.Tensor = fake_mode[pos_fake_idx]
        pos_fake += torch.randn_like(pos_fake) * pos_fake_std / 2.0
        # real position
        n_masked = ligand_masked_pos.size(0)
        p = np.ones(n_masked, dtype=np.float32) / n_masked
        pos_real_idx: NDArray[np.intp] = np.random.choice(np.arange(n_masked), size=num_real_pos, p=p)
        pos_real: torch.Tensor = ligand_masked_pos[pos_real_idx]
        pos_real += torch.randn_like(pos_real) * pos_real_std
    else:
        # fake position
        fake_mode = ligand_context_pos[data.ligand_frontier]
        n_fake = fake_mode.size(0)
        p = np.ones(n_fake, dtype=np.float32) / n_fake
        pos_fake_idx = np.random.choice(np.arange(n_fake), size=num_fake_pos, p=p)
        pos_fake = fake_mode[pos_fake_idx]
        pos_fake += torch.randn_like(pos_fake) * pos_fake_std / 2.0
        # real position
        n_masked = ligand_masked_pos.size(0)
        p = np.ones(n_masked, dtype=np.float32) / n_masked
        pos_real_idx = np.random.choice(np.arange(n_masked), size=num_real_pos - 1, p=p)
        pos_real = ligand_masked_pos[pos_real_idx]
        pos_real += torch.randn_like(pos_real) * pos_real_std
        pos_real = torch.cat([pos_real, data.y_pos], dim=0)

    data.pos_fake = pos_fake
    pos_fake_knn_edge_idx = knn(x=data.cpx_pos, y=pos_fake, k=k, num_workers=num_workers)
    data.pos_fake_knn_edge_idx_0, data.pos_fake_knn_edge_idx_1 = (
        pos_fake_knn_edge_idx[1],
        pos_fake_knn_edge_idx[0],
    )

    data.pos_real = pos_real
    pos_real_knn_edge_idx = knn(x=data.cpx_pos, y=pos_real, k=k, num_workers=num_workers)
    data.pos_real_knn_edge_idx_0, data.pos_real_knn_edge_idx_1 = (
        pos_real_knn_edge_idx[1],
        pos_real_knn_edge_idx[0],
    )
    return data


def get_complex_graph(
    data: ComplexData,
    len_ligand_ctx: int,
    len_compose: int,
    num_workers: int,
    graph_type: GraphType = GraphType.KNN,
    knn: int = 16,
    radius: float = 10.0,
) -> ComplexData:
    """Build a composed kNN graph and annotate edges that are true ligand bonds.

    The composed node set is expected to be ordered as:

    1) ligand context atoms (``len_ligand_ctx`` nodes), then
    2) protein atoms (remaining nodes up to ``len_compose``).

    The graph is constructed with :func:`torch_geometric.nn.pool.knn_graph` on
    ``data.cpx_pos``. Any kNN edge among ligand-context nodes that corresponds
    to a true ligand bond (as defined by ``data.ligand_context_bond_index``) is
    assigned that bond's type; all other edges are assigned type 0 ("no bond").

    Notes:
        - The current implementation always uses kNN (``knn_graph``) and does
          not use the ``graph_type`` or ``radius`` parameters.
        - Only ligand-context bonds are annotated; protein bonds are ignored in
          this codepath (see :func:`get_complex_graph_`).

    Required input attributes on ``data``:
        - ``cpx_pos``: composed positions, shape ``(len_compose, 3)``.
        - ``ligand_context_bond_index`` / ``ligand_context_bond_type``: context
          bond edges/types (relabelled to ``0..len_ligand_ctx-1``).

    Added attributes:
        - ``cpx_edge_index``: composed kNN edge index.
        - ``cpx_edge_type``: integer edge types (0=no bond, 1/2/3=bond type).
        - ``cpx_edge_feature``: one-hot edge features, shape ``(E, 4)``.

    Args:
        data: PyG data object to mutate.
        len_ligand_ctx: Number of ligand context nodes in the composed ordering.
        len_compose: Total number of composed nodes.
        num_workers: Worker count for kNN construction.
        graph_type: Graph construction type (currently unused; always uses kNN).
        knn: Number of neighbors per node in kNN graph construction.
        radius: Unused; kept for API compatibility.

    Returns:
        The mutated ``data`` object.
    """
    data.cpx_edge_index = knn_graph(
        data.cpx_pos,
        knn,
        flow="source_to_target",
        num_workers=num_workers,
    )

    id_cpx_edge = data.cpx_edge_index[0] * len_compose + data.cpx_edge_index[1]
    bond_id = data.ligand_context_bond_index[0] * len_compose + data.ligand_context_bond_index[1]

    data.cpx_edge_type = torch.zeros(data.cpx_edge_index.size(1), dtype=torch.long)
    if bond_id.numel() > 0:
        perm = bond_id.argsort()
        bond_id_sorted = bond_id[perm]
        bond_type_sorted = data.ligand_context_bond_type[perm]
        idx = torch.searchsorted(bond_id_sorted, id_cpx_edge)
        valid = idx < bond_id_sorted.numel()
        idx = idx.clamp_max(bond_id_sorted.numel() - 1)
        match = valid & (bond_id_sorted[idx] == id_cpx_edge)
        data.cpx_edge_type[match] = bond_type_sorted[idx[match]]

    data.cpx_edge_feature = F.one_hot(data.cpx_edge_type, num_classes=4).long()
    return data


def get_knn_graph(
    pos: torch.Tensor,
    k: int = 16,
    edge_feat: torch.Tensor | None = None,
    edge_feat_index: torch.Tensor | None = None,
    *,
    num_workers: int,
    graph_type: GraphType = GraphType.KNN,
    radius: float = 5.5,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Construct a kNN (or radius) graph and optionally look up edge attributes.

    When ``edge_feat`` and ``edge_feat_index`` are provided, this function
    creates a dense adjacency matrix to map each returned edge to an attribute
    value. This is convenient for small/medium graphs but may be memory-heavy
    for large ``pos.size(0)``.

    Args:
        pos: Node positions, shape ``(N, 3)``.
        k: Number of nearest neighbors when ``graph_type=GraphType.KNN``.
        edge_feat: Optional edge attribute values, shape ``(E_attr,)``.
        edge_feat_index: Optional edge indices for attributes, shape
            ``(2, E_attr)`` with indices in ``0..N-1``.
        num_workers: Worker count for PyG neighbor search.
        graph_type: Graph construction type (``GraphType.KNN`` or ``GraphType.RAD``).
        radius: Cutoff when ``graph_type=GraphType.RAD``.

    Returns:
        A tuple ``(edge_index, edge_type)`` where:
          - ``edge_index`` is the constructed graph edge index, shape ``(2, E)``.
          - ``edge_type`` is either `None` (if no attributes were provided) or a
            tensor of shape ``(E,)`` containing attribute values for each edge.
    """
    cpx_edge_index: torch.Tensor
    if graph_type == GraphType.RAD:
        cpx_edge_index = radius_graph(pos, radius, flow="source_to_target", num_workers=num_workers)
    else:  # graph_type == GraphType.KNN
        cpx_edge_index = knn_graph(pos, k, flow="source_to_target", num_workers=num_workers).long()

    cpx_edge_type: torch.Tensor | None
    if isinstance(edge_feat, torch.Tensor) and isinstance(edge_feat_index, torch.Tensor):
        adj_feat_mat = torch.zeros([pos.size(0), pos.size(0)], dtype=torch.long)
        adj_feat_mat[edge_feat_index[0], edge_feat_index[1]] = edge_feat
        cpx_edge_type = adj_feat_mat[cpx_edge_index[0], cpx_edge_index[1]]
    else:
        cpx_edge_type = None
    return cpx_edge_index, cpx_edge_type


def get_complex_graph_(
    data: ComplexData,
    knn: int = 16,
    *,
    num_workers: int,
    graph_type: GraphType = GraphType.KNN,
    radius: float = 5.5,
) -> ComplexData:
    """Build a composed graph with ligand and protein bond annotations.

    This constructs a kNN/radius graph on ``data.cpx_pos`` and annotates each
    returned edge with a bond type if it corresponds to either:

    - a ligand-context bond (from ``data.ligand_context_bond_*``), or
    - a protein bond (from ``data.protein_bond_*``), offset into composed space.

    Required input attributes on ``data``:
        - ``cpx_pos``: composed positions.
        - ``ligand_context_pos``: used to compute the offset for protein indices.
        - ``ligand_context_bond_index`` / ``ligand_context_bond_type``.
        - ``protein_bond_index`` / ``protein_bond_type``.

    Added attributes:
        - ``cpx_edge_index`` / ``cpx_edge_type`` / ``cpx_edge_feature``.

    Args:
        data: PyG data object to mutate.
        knn: Number of nearest neighbors for kNN.
        num_workers: Worker count for PyG neighbor search.
        graph_type: Graph construction type (``GraphType.KNN`` or ``GraphType.RAD``).
        radius: Cutoff when ``graph_type=GraphType.RAD``.

    Returns:
        The mutated ``data`` object.
    """
    edge_feat: torch.Tensor = torch.cat([data.ligand_context_bond_type, data.protein_bond_type]).long()
    edge_feat_index: torch.Tensor = torch.cat(
        [data.ligand_context_bond_index, data.protein_bond_index + data.ligand_context_pos.size(0)], dim=1
    ).long()
    knn_edge_index, knn_edge_type = get_knn_graph(
        data.cpx_pos,
        k=knn,
        edge_feat=edge_feat,
        edge_feat_index=edge_feat_index,
        graph_type=graph_type,
        num_workers=num_workers,
        radius=radius,
    )
    assert knn_edge_type is not None
    data.cpx_edge_index = knn_edge_index
    data.cpx_edge_type = knn_edge_type
    data.cpx_edge_feature = F.one_hot(knn_edge_type, num_classes=4)
    return data


def sample_edge_with_radius(data: ComplexData, r: float = 4.0, *, num_workers: int) -> ComplexData:
    """Construct query→context edges within a radius and label true ligand bonds.

    This helper builds a set of edges between the current query position
    ``data.y_pos`` (typically a single next-atom position) and ligand-context
    atoms within radius ``r``. Each such edge is assigned a label:

    - ``0`` if there is no bond between the query atom and the context atom
      in the *original* ligand graph.
    - ``1/2/3`` if the original ligand has a bond of that type between the
      current query atom (``data.masked_idx[0]``) and that context atom.

    Required input attributes on ``data``:
        - ``y_pos``: query position(s), shape ``(N_query, 3)``.
        - ``ligand_context_pos``: context positions, shape ``(N_ctx, 3)``.
        - ``context_idx`` / ``masked_idx``: indices into full ligand atom space.
        - ``ligand_bond_index`` / ``ligand_bond_type``: full ligand bond graph.

    Added attributes:
        - ``edge_query_index_0`` / ``edge_query_index_1``: edge indices where
          ``edge_query_index_0`` indexes query positions (``y_pos`` rows) and
          ``edge_query_index_1`` indexes composed atoms (rows of ``cpx_pos``),
          obtained by mapping ctx indices through ``idx_ligand_ctx_in_cpx``.
        - ``edge_label``: integer bond-type labels for each queried edge.

    Args:
        data: PyG data object to mutate.
        r: Radius cutoff for candidate edges.
        num_workers: Worker count for PyG neighbor search.

    Returns:
        The mutated ``data`` object.
    """
    y_pos: torch.Tensor = data.y_pos
    ligand_context_pos: torch.Tensor = data.ligand_context_pos
    context_idx: torch.Tensor = data.context_idx
    masked_idx: torch.Tensor = data.masked_idx
    ligand_bond_index: torch.Tensor = data.ligand_bond_index
    ligand_bond_type: torch.Tensor = data.ligand_bond_type
    idx_ctx_in_cpx: torch.Tensor = data.idx_ligand_ctx_in_cpx
    assert idx_ctx_in_cpx.numel() == ligand_context_pos.size(0)
    # select the atoms whose distance < r between pos_query as edge samples
    edge_index_radius: torch.Tensor = radius(ligand_context_pos, y_pos, r=r, num_workers=num_workers)
    # get the labels of edge samples
    # Vectorized membership checks: check if bond edges connect masked_idx[0] to context nodes within radius
    context_nodes_in_radius = context_idx[edge_index_radius[1]]
    # Check if source nodes equal masked_idx[0] and destination nodes are in context_nodes_in_radius
    mask_i = ligand_bond_index[0] == masked_idx[0]
    mask_j = torch.isin(ligand_bond_index[1], context_nodes_in_radius)
    mask = mask_i & mask_j
    new_idx_1 = torch.nonzero((ligand_bond_index[:, mask][1].view(-1, 1) == context_idx).any(0)).view(-1)
    real_bond_type_in_edge_index_radius = torch.nonzero(
        (new_idx_1.view(-1, 1) == edge_index_radius[1]).any(0)
    ).view(-1)
    edge_label = torch.zeros(edge_index_radius.size(1), dtype=torch.long)
    edge_label[real_bond_type_in_edge_index_radius] = ligand_bond_type[mask]
    data.edge_query_index_0 = edge_index_radius[0]
    data.edge_query_index_1 = idx_ctx_in_cpx[edge_index_radius[1]]
    data.edge_label = edge_label
    return data


def get_tri_edges(
    edge_index_query_ctx: torch.Tensor,
    pos_query: torch.Tensor,
    idx_ligand_ctx_in_cpx: torch.Tensor,
    ligand_bond_index: torch.Tensor,
    ligand_bond_type: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build "triangle-edge" features for attention over queried edges.

    Given a set of queried edges from query positions to ligand-context nodes
    (``edge_index_query``), this function enumerates *all ordered pairs* of
    queried edges incident to the same query node. For each pair, it creates:

    - indices into the queried-edge list (used to gather edge embeddings), and
    - a pair of context-node indices (a "triangle edge") with a one-hot feature
      describing the relation between those two context nodes in the ligand
      context bond graph.

    The relation types are encoded as a 5-way one-hot over values
    ``{-1, 0, 1, 2, 3}``, where:

    - ``-1``: same context node (diagonal).
    - ``0``: distinct nodes with no bond in the context graph.
    - ``1/2/3``: bond types from ``ligand_bond_type``.

    Complexity:
        If a query node has degree ``d`` (number of queried edges incident to
        it), this enumerates ``d²`` pairs. Total work is ``Σ d_i²``.

    Args:
        edge_index_query_ctx: Query→context edge index of shape ``(2, E_query)``,
            where ``edge_index_query_ctx[0]`` indexes query nodes (rows of
            ``pos_query``) and ``edge_index_query_ctx[1]`` indexes ctx nodes.
        pos_query: Query positions, shape ``(N_query, 3)``. Used for device
            placement only.
        idx_ligand_ctx_in_cpx: Mapping from ctx indices to composed indices.
            Used for ``N_ctx`` and for mapping triangle endpoints to cpx space.
        ligand_bond_index: Context bond edge index, shape ``(2, E_ctx)`` with
            indices in ``0..N_ctx-1``.
        ligand_bond_type: Context bond types, shape ``(E_ctx,)`` with values in
            ``{1, 2, 3}``.

    Returns:
        A tuple ``(index_real_cps_edge_for_atten, tri_edge_index, tri_edge_feat)``:
          - ``index_real_cps_edge_for_atten``: stacked indices into the queried
            edge list, shape ``(2, E_att)``.
          - ``tri_edge_index``: stacked composed-node pairs in source→target order,
            shape ``(2, E_att)``.
          - ``tri_edge_feat``: one-hot relation features, shape ``(E_att, 5)``.
    """
    row, col_ctx = edge_index_query_ctx
    acc_num_edges = 0
    index_real_cps_edge_i_list: list[torch.Tensor] = []
    index_real_cps_edge_j_list: list[torch.Tensor] = []
    for node in range(pos_query.size(0)):
        num_edges = int((row == node).sum().item())
        index_edge_i = torch.arange(num_edges, dtype=torch.long) + acc_num_edges
        index_edge_i, index_edge_j = torch.meshgrid(index_edge_i, index_edge_i, indexing="ij")
        index_edge_i, index_edge_j = index_edge_i.flatten(), index_edge_j.flatten()
        index_real_cps_edge_i_list.append(index_edge_i)
        index_real_cps_edge_j_list.append(index_edge_j)
        acc_num_edges += num_edges
    index_real_cps_edge_i = torch.cat(index_real_cps_edge_i_list, dim=0).to(pos_query.device)
    index_real_cps_edge_j = torch.cat(index_real_cps_edge_j_list, dim=0).to(pos_query.device)

    node_a_ctx = col_ctx[index_real_cps_edge_i]
    node_b_ctx = col_ctx[index_real_cps_edge_j]
    n_context = int(idx_ligand_ctx_in_cpx.numel())
    adj_mat = torch.zeros([n_context, n_context], dtype=torch.long) - torch.eye(n_context, dtype=torch.long)
    adj_mat = adj_mat.to(ligand_bond_index.device)
    adj_mat[ligand_bond_index[0], ligand_bond_index[1]] = ligand_bond_type
    tri_edge_type = adj_mat[node_b_ctx, node_a_ctx]
    tri_edge_feat = (
        tri_edge_type.view([-1, 1]) == torch.tensor([[-1, 0, 1, 2, 3]]).to(tri_edge_type.device)
    ).long()

    index_real_cps_edge_for_atten = torch.stack(
        [
            index_real_cps_edge_i,
            index_real_cps_edge_j,
        ],
        dim=0,
    )
    tri_edge_index = torch.stack(
        [
            idx_ligand_ctx_in_cpx[node_b_ctx],
            idx_ligand_ctx_in_cpx[node_a_ctx],
        ],
        dim=0,
    )
    return index_real_cps_edge_for_atten, tri_edge_index, tri_edge_feat
