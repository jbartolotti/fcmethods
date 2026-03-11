"""
Graph analysis utilities for ROI-level functional connectivity matrices.

Implements CONN-style graph construction from ROI-to-ROI correlation matrices:
- nodes = ROIs
- edges = supra-threshold connections in undirected binary adjacency matrices

Includes node-level and network-level graph metrics across threshold ranges,
and AUC summaries across thresholds.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse.csgraph import shortest_path, connected_components


def build_adjacency_from_corrmat(
    matrix: np.ndarray,
    threshold_mode: str = "cost",
    threshold_value: float = 0.10,
    positive_only: bool = True,
) -> np.ndarray:
    """
    Build an undirected binary adjacency matrix from a correlation matrix.

    Parameters
    ----------
    matrix : np.ndarray
        Symmetric ROI-to-ROI matrix (e.g., Fisher-z transformed correlations)
    threshold_mode : str, optional
        Thresholding mode: "cost" (proportional) or "absolute". Default: "cost"
    threshold_value : float, optional
        For "cost", fraction of total possible undirected edges to retain (0..1).
        For "absolute", edge cutoff in matrix units. Default: 0.10
    positive_only : bool, optional
        If True, only positive edges are eligible to be kept. Default: True

    Returns
    -------
    adjacency : np.ndarray
        Binary symmetric adjacency matrix (n_rois x n_rois), diagonal zero.
    """
    m = np.asarray(matrix, dtype=float).copy()
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError("Input matrix must be square (n_rois x n_rois).")

    n = m.shape[0]
    np.fill_diagonal(m, np.nan)

    triu_i, triu_j = np.triu_indices(n, k=1)
    edge_values = m[triu_i, triu_j]

    valid_mask = ~np.isnan(edge_values)
    if positive_only:
        valid_mask &= edge_values > 0

    adjacency = np.zeros((n, n), dtype=int)

    if not np.any(valid_mask):
        return adjacency

    valid_indices = np.where(valid_mask)[0]
    valid_values = edge_values[valid_mask]

    if threshold_mode == "cost":
        if threshold_value < 0 or threshold_value > 1:
            raise ValueError("Cost threshold must be within [0, 1].")

        n_possible = n * (n - 1) // 2
        n_keep = int(np.floor(threshold_value * n_possible))
        if threshold_value > 0 and n_keep == 0:
            n_keep = 1
        n_keep = min(n_keep, len(valid_indices))

        if n_keep > 0:
            order = np.argsort(valid_values)[::-1]
            keep_local = order[:n_keep]
            keep_global = valid_indices[keep_local]
            i_keep = triu_i[keep_global]
            j_keep = triu_j[keep_global]
            adjacency[i_keep, j_keep] = 1
            adjacency[j_keep, i_keep] = 1

    elif threshold_mode == "absolute":
        keep_local = np.where(valid_values >= threshold_value)[0]
        if keep_local.size > 0:
            keep_global = valid_indices[keep_local]
            i_keep = triu_i[keep_global]
            j_keep = triu_j[keep_global]
            adjacency[i_keep, j_keep] = 1
            adjacency[j_keep, i_keep] = 1
    else:
        raise ValueError("threshold_mode must be 'cost' or 'absolute'.")

    np.fill_diagonal(adjacency, 0)
    return adjacency


def _brandes_betweenness_unweighted_undirected(adjacency: np.ndarray) -> np.ndarray:
    """Compute node betweenness centrality using Brandes algorithm."""
    n = adjacency.shape[0]
    neighbors = [np.where(adjacency[v] > 0)[0].tolist() for v in range(n)]
    bc = np.zeros(n, dtype=float)

    for s in range(n):
        stack = []
        pred = [[] for _ in range(n)]
        sigma = np.zeros(n, dtype=float)
        sigma[s] = 1.0
        dist = -np.ones(n, dtype=int)
        dist[s] = 0

        queue = [s]
        q_head = 0
        while q_head < len(queue):
            v = queue[q_head]
            q_head += 1
            stack.append(v)
            for w in neighbors[v]:
                if dist[w] < 0:
                    queue.append(w)
                    dist[w] = dist[v] + 1
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    pred[w].append(v)

        delta = np.zeros(n, dtype=float)
        while stack:
            w = stack.pop()
            if sigma[w] > 0:
                coeff = (1.0 + delta[w]) / sigma[w]
                for v in pred[w]:
                    delta[v] += sigma[v] * coeff
            if w != s:
                bc[w] += delta[w]

    # Undirected graph normalization: divide by 2
    bc *= 0.5

    # Normalize to [0, 1] for comparability
    if n > 2:
        bc /= ((n - 1) * (n - 2) / 2.0)

    return bc


def compute_node_graph_metrics(adjacency: np.ndarray) -> pd.DataFrame:
    """
    Compute ROI-level graph metrics from an undirected binary adjacency matrix.

    Returns
    -------
    metrics_df : pd.DataFrame
        Columns: degree, cost, avg_path_distance, clustering_coefficient,
        global_efficiency, local_efficiency, betweenness_centrality
    """
    a = np.asarray(adjacency, dtype=int)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("Adjacency must be square.")

    n = a.shape[0]
    if n < 2:
        raise ValueError("Adjacency must contain at least 2 nodes.")

    degree = a.sum(axis=1).astype(float)
    cost = degree / float(n - 1)

    dist = shortest_path(a, directed=False, unweighted=True)
    np.fill_diagonal(dist, np.nan)

    # Node average path distance over finite reachable nodes
    avg_path_distance = np.full(n, np.nan, dtype=float)
    for i in range(n):
        finite_vals = dist[i, np.isfinite(dist[i])]
        if finite_vals.size > 0:
            avg_path_distance[i] = np.nanmean(finite_vals)

    # Node global efficiency: average inverse distance to all other nodes
    inv_dist = np.where(np.isfinite(dist), 1.0 / dist, 0.0)
    inv_dist = np.where(np.isnan(inv_dist), 0.0, inv_dist)
    global_efficiency = inv_dist.sum(axis=1) / float(n - 1)

    # Node clustering coefficient
    clustering = np.zeros(n, dtype=float)
    local_efficiency = np.zeros(n, dtype=float)

    for i in range(n):
        neighbors = np.where(a[i] > 0)[0]
        k = len(neighbors)

        if k < 2:
            clustering[i] = 0.0
            local_efficiency[i] = 0.0
            continue

        sub = a[np.ix_(neighbors, neighbors)]
        e_sub = sub.sum() / 2.0
        clustering[i] = (2.0 * e_sub) / (k * (k - 1))

        # Local efficiency = global efficiency of neighboring subgraph
        sub_dist = shortest_path(sub, directed=False, unweighted=True)
        np.fill_diagonal(sub_dist, np.nan)
        sub_inv = np.where(np.isfinite(sub_dist), 1.0 / sub_dist, 0.0)
        sub_inv = np.where(np.isnan(sub_inv), 0.0, sub_inv)
        local_efficiency[i] = sub_inv.sum() / (k * (k - 1))

    betweenness = _brandes_betweenness_unweighted_undirected(a)

    return pd.DataFrame(
        {
            "degree": degree,
            "cost": cost,
            "avg_path_distance": avg_path_distance,
            "clustering_coefficient": clustering,
            "global_efficiency": global_efficiency,
            "local_efficiency": local_efficiency,
            "betweenness_centrality": betweenness,
        }
    )


def compute_network_graph_metrics(adjacency: np.ndarray, node_metrics: Optional[pd.DataFrame] = None) -> Dict[str, float]:
    """
    Compute network-level graph metrics from an undirected binary adjacency matrix.
    """
    a = np.asarray(adjacency, dtype=int)
    n = a.shape[0]

    if node_metrics is None:
        node_metrics = compute_node_graph_metrics(a)

    n_edges = int(a.sum() // 2)
    n_possible = int(n * (n - 1) // 2)
    density = (n_edges / n_possible) if n_possible > 0 else np.nan

    n_components, _ = connected_components(a, directed=False, return_labels=True)

    graph = nx.from_numpy_array(a)
    if graph.number_of_edges() == 0:
        modularity = 0.0
    else:
        communities = list(nx.algorithms.community.greedy_modularity_communities(graph))
        modularity = float(nx.algorithms.community.modularity(graph, communities))

    return {
        "n_nodes": int(n),
        "n_edges": n_edges,
        "density": float(density),
        "n_components": int(n_components),
        "degree": float(np.nanmean(node_metrics["degree"])),
        "cost": float(np.nanmean(node_metrics["cost"])),
        "avg_path_distance": float(np.nanmean(node_metrics["avg_path_distance"])),
        "clustering_coefficient": float(np.nanmean(node_metrics["clustering_coefficient"])),
        "global_efficiency": float(np.nanmean(node_metrics["global_efficiency"])),
        "local_efficiency": float(np.nanmean(node_metrics["local_efficiency"])),
        "betweenness_centrality": float(np.nanmean(node_metrics["betweenness_centrality"])),
        "modularity": modularity,
    }


def compute_auc_by_group(
    df: pd.DataFrame,
    threshold_col: str,
    metric_cols: List[str],
    group_cols: List[str],
) -> pd.DataFrame:
    """
    Compute AUC across thresholds for selected metric columns.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format metrics table with threshold column.
    threshold_col : str
        Column containing threshold values (numeric).
    metric_cols : list
        Metric columns to integrate.
    group_cols : list
        Grouping columns defining one trajectory across thresholds.
    """
    if df.empty:
        return pd.DataFrame(columns=group_cols + metric_cols)

    out_rows = []
    for keys, sub_df in df.groupby(group_cols, dropna=False):
        sub = sub_df.sort_values(threshold_col)
        x = sub[threshold_col].values.astype(float)

        if len(metric_cols) == 0:
            continue

        row = {}
        if not isinstance(keys, tuple):
            keys = (keys,)
        for col, val in zip(group_cols, keys):
            row[col] = val

        for metric in metric_cols:
            y = sub[metric].values.astype(float)
            if len(x) < 2:
                row[metric] = np.nan
            else:
                row[metric] = float(np.trapz(y, x))
        out_rows.append(row)

    return pd.DataFrame(out_rows)


def save_graph_outputs(
    output_dir: Path,
    node_df: pd.DataFrame,
    network_df: pd.DataFrame,
    node_auc_df: Optional[pd.DataFrame],
    network_auc_df: Optional[pd.DataFrame],
    metadata: Dict,
) -> Dict[str, Path]:
    """Save graph analysis outputs as TSV + JSON sidecars."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = {}

    node_tsv = output_dir / "graphmetrics_desc-node.tsv"
    node_json = output_dir / "graphmetrics_desc-node.json"
    node_df.to_csv(node_tsv, sep="\t", index=False)
    with open(node_json, "w") as f:
        json.dump({**metadata, "Level": "node"}, f, indent=2)
    outputs["node_tsv"] = node_tsv
    outputs["node_json"] = node_json

    network_tsv = output_dir / "graphmetrics_desc-network.tsv"
    network_json = output_dir / "graphmetrics_desc-network.json"
    network_df.to_csv(network_tsv, sep="\t", index=False)
    with open(network_json, "w") as f:
        json.dump({**metadata, "Level": "network"}, f, indent=2)
    outputs["network_tsv"] = network_tsv
    outputs["network_json"] = network_json

    if node_auc_df is not None and not node_auc_df.empty:
        node_auc_tsv = output_dir / "graphmetrics_desc-nodeAUC.tsv"
        node_auc_json = output_dir / "graphmetrics_desc-nodeAUC.json"
        node_auc_df.to_csv(node_auc_tsv, sep="\t", index=False)
        with open(node_auc_json, "w") as f:
            json.dump({**metadata, "Level": "node", "Summary": "AUC"}, f, indent=2)
        outputs["node_auc_tsv"] = node_auc_tsv
        outputs["node_auc_json"] = node_auc_json

    if network_auc_df is not None and not network_auc_df.empty:
        network_auc_tsv = output_dir / "graphmetrics_desc-networkAUC.tsv"
        network_auc_json = output_dir / "graphmetrics_desc-networkAUC.json"
        network_auc_df.to_csv(network_auc_tsv, sep="\t", index=False)
        with open(network_auc_json, "w") as f:
            json.dump({**metadata, "Level": "network", "Summary": "AUC"}, f, indent=2)
        outputs["network_auc_tsv"] = network_auc_tsv
        outputs["network_auc_json"] = network_auc_json

    return outputs
