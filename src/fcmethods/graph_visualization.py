"""
Visualization utilities for graph metrics outputs.
"""

from pathlib import Path
from typing import Dict, List, Optional
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist

from .graph_analysis import build_adjacency_from_corrmat


GRAPH_METRIC_COLUMNS = [
    "degree",
    "cost",
    "avg_path_distance",
    "clustering_coefficient",
    "global_efficiency",
    "local_efficiency",
    "betweenness_centrality",
    "modularity",
]

EDGE_PREVALENCE_LAYOUT_TYPES = [
    "circle",
    "clustered_circle",
    "community",
    "force_fixed",
]


def _load_graph_table(graph_dir: Path, level: str) -> pd.DataFrame:
    """Load normalized AUC first, then AUC, then threshold-wise table."""
    graph_dir = Path(graph_dir)

    auc_norm_path = graph_dir / f"graphmetrics_desc-{level}AUCnorm.tsv"
    auc_path = graph_dir / f"graphmetrics_desc-{level}AUC.tsv"
    raw_path = graph_dir / f"graphmetrics_desc-{level}.tsv"

    if auc_norm_path.exists():
        return pd.read_csv(auc_norm_path, sep="\t")
    if auc_path.exists():
        return pd.read_csv(auc_path, sep="\t")
    if raw_path.exists():
        return pd.read_csv(raw_path, sep="\t")

    raise FileNotFoundError(f"Could not find {level} metrics table in {graph_dir}")


def _dot_box_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    output_path: Path,
    order: Optional[List[str]] = None,
    participant_col: Optional[str] = None,
    condition_col: Optional[str] = None,
    condition_order: Optional[List[str]] = None,
    condition_display_names: Optional[Dict[str, str]] = None,
    group_col: Optional[str] = None,
    figsize: tuple = (8, 6),
    dpi: int = 150,
) -> None:
    """Create a dot+box plot and save figure."""
    if df.empty:
        return

    required_cols = [x_col, y_col]
    for optional_col in [participant_col, condition_col, group_col]:
        if optional_col is not None:
            required_cols.append(optional_col)

    plot_df = df[required_cols].dropna(subset=[x_col, y_col]).copy()
    if plot_df.empty:
        return

    if order is None:
        categories = list(pd.unique(plot_df[x_col]))
    else:
        categories = [cat for cat in order if cat in set(plot_df[x_col])]

    if not categories:
        return

    y_data = [plot_df.loc[plot_df[x_col] == cat, y_col].values for cat in categories]
    category_to_pos = {cat: idx for idx, cat in enumerate(categories)}

    fig, ax = plt.subplots(figsize=figsize)

    # Paired lines (intervention vs control within participant)
    if (
        participant_col is not None
        and condition_col is not None
        and condition_order is not None
        and len(condition_order) >= 2
    ):
        condition_display_names = condition_display_names or {}
        if group_col is None:
            line_group_cols = [participant_col]
        else:
            line_group_cols = [participant_col, group_col]

        for _, sub_df in plot_df.groupby(line_group_cols, dropna=False):
            y_by_cond = {}
            for cond in condition_order:
                cond_df = sub_df[sub_df[condition_col] == cond]
                if cond_df.empty:
                    continue

                cond_label = condition_display_names.get(cond, cond)
                if group_col is not None and group_col in sub_df.columns and not sub_df[group_col].isna().all():
                    grp = str(sub_df[group_col].iloc[0])
                    x_label = f"{grp} {cond_label}"
                else:
                    x_label = cond_label

                if x_label not in category_to_pos:
                    continue

                y_by_cond[cond] = float(cond_df[y_col].mean())

            cond_a = condition_order[0]
            cond_b = condition_order[1]
            if cond_a in y_by_cond and cond_b in y_by_cond:
                cond_a_label = condition_display_names.get(cond_a, cond_a)
                cond_b_label = condition_display_names.get(cond_b, cond_b)

                if group_col is not None and group_col in sub_df.columns and not sub_df[group_col].isna().all():
                    grp = str(sub_df[group_col].iloc[0])
                    x_a_label = f"{grp} {cond_a_label}"
                    x_b_label = f"{grp} {cond_b_label}"
                else:
                    x_a_label = cond_a_label
                    x_b_label = cond_b_label

                if x_a_label in category_to_pos and x_b_label in category_to_pos:
                    x_a = category_to_pos[x_a_label]
                    x_b = category_to_pos[x_b_label]
                    y_a = y_by_cond[cond_a]
                    y_b = y_by_cond[cond_b]

                    # Light blue if intervention > control, light red otherwise
                    if y_a > y_b:
                        line_color = "#9BBFE5"
                    else:
                        line_color = "#E6B0B0"

                    ax.plot(
                        [x_a, x_b],
                        [y_a, y_b],
                        color=line_color,
                        linewidth=1.0,
                        alpha=0.45,
                        zorder=1,
                    )

    # Box-and-whisker
    ax.boxplot(
        y_data,
        positions=np.arange(len(categories)),
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        boxprops={"facecolor": "#D9D9D9", "edgecolor": "black", "linewidth": 1.2},
        medianprops={"color": "black", "linewidth": 1.5},
        whiskerprops={"color": "black", "linewidth": 1.0},
        capprops={"color": "black", "linewidth": 1.0},
    )

    # Dotplot (participant-level points)
    rng = np.random.default_rng(42)
    for i, cat in enumerate(categories):
        vals = plot_df.loc[plot_df[x_col] == cat, y_col].values
        jitter = rng.uniform(-0.14, 0.14, size=len(vals))
        ax.scatter(
            np.full(len(vals), i) + jitter,
            vals,
            s=28,
            alpha=0.80,
            color="#1F77B4",
            edgecolor="white",
            linewidth=0.4,
            zorder=3,
        )

    ax.set_xticks(np.arange(len(categories)))
    ax.set_xticklabels(categories, rotation=25, ha="right")
    ax.set_ylabel(y_col)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _get_roi_labels_for_matrix(output_root: Path, matrix_type: str, subjects: List[str], n_rois: int) -> List[str]:
    """Infer ROI labels from corrmat JSON sidecars."""
    for subject_id in subjects:
        json_path = output_root / f"sub-{subject_id}" / f"corrmat_{matrix_type}.json"
        if json_path.exists():
            try:
                with open(json_path, "r") as f:
                    meta = json.load(f)
                rois = meta.get("ROIs")
                if isinstance(rois, list) and len(rois) == n_rois:
                    return [str(r) for r in rois]
            except Exception:
                continue
    return [f"ROI_{i + 1}" for i in range(n_rois)]


def _plot_prevalence_network(ax, prevalence: np.ndarray, roi_labels: List[str], title: str) -> None:
    """Plot network where edge darkness scales with prevalence (0..1)."""
    n = prevalence.shape[0]
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)

    for i in range(n):
        for j in range(i + 1, n):
            w = float(prevalence[i, j])
            if w <= 0:
                continue
            ax.plot(
                [x[i], x[j]],
                [y[i], y[j]],
                color="black",
                alpha=min(0.95, max(0.08, w)),
                linewidth=0.4 + 2.8 * w,
                zorder=1,
            )

    ax.scatter(x, y, s=65, color="white", edgecolor="black", linewidth=1.0, zorder=3)

    # Label ROIs (small font to reduce clutter)
    for i in range(n):
        ax.text(x[i] * 1.12, y[i] * 1.12, roi_labels[i], fontsize=5.5, ha="center", va="center")

    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.axis("off")
    ax.set_aspect("equal")


def _compute_mean_prevalence_matrix(
    output_root: Path,
    subjects: List[str],
    matrix_type: str,
    threshold_mode: str,
    thresholds: List[float],
    positive_only: bool,
) -> np.ndarray:
    """Compute a mean prevalence matrix across thresholds for layout estimation."""
    prevalence_accumulator = None
    n_thresholds_used = 0

    for thr in thresholds:
        threshold_sum = None
        n_used = 0
        for sid in subjects:
            mat_path = output_root / f"sub-{sid}" / f"corrmat_{matrix_type}.npy"
            if not mat_path.exists():
                continue
            mat = np.load(mat_path)
            if threshold_sum is None:
                threshold_sum = np.zeros_like(mat, dtype=float)
            if mat.shape != threshold_sum.shape:
                continue
            adj = build_adjacency_from_corrmat(
                matrix=mat,
                threshold_mode=threshold_mode,
                threshold_value=thr,
                positive_only=positive_only,
            )
            threshold_sum += adj
            n_used += 1

        if threshold_sum is not None and n_used > 0:
            prevalence = threshold_sum / float(n_used)
            if prevalence_accumulator is None:
                prevalence_accumulator = np.zeros_like(prevalence, dtype=float)
            prevalence_accumulator += prevalence
            n_thresholds_used += 1

    if prevalence_accumulator is None or n_thresholds_used == 0:
        raise RuntimeError(f"Could not compute mean prevalence matrix for {matrix_type}")

    return prevalence_accumulator / float(n_thresholds_used)


def _compute_circle_positions(n_rois: int, order: Optional[List[int]] = None) -> Dict[int, np.ndarray]:
    """Return circular node positions using optional node order."""
    if order is None:
        order = list(range(n_rois))
    angles = np.linspace(0, 2 * np.pi, n_rois, endpoint=False)
    pos = {}
    for angle, node_idx in zip(angles, order):
        pos[node_idx] = np.array([np.cos(angle), np.sin(angle)])
    return pos


def _compute_clustered_circle_order(prevalence: np.ndarray) -> List[int]:
    """Order nodes on a circle using hierarchical clustering of connectivity profiles."""
    n_rois = prevalence.shape[0]
    if n_rois <= 2:
        return list(range(n_rois))

    profiles = prevalence.copy()
    np.fill_diagonal(profiles, 0.0)

    try:
        distances = pdist(profiles, metric="euclidean")
        if np.allclose(distances, 0):
            return list(range(n_rois))
        linkage_matrix = linkage(distances, method="average")
        return leaves_list(linkage_matrix).tolist()
    except Exception:
        return list(range(n_rois))


def _compute_force_fixed_positions(prevalence: np.ndarray) -> Dict[int, np.ndarray]:
    """Compute a single force-directed layout from weighted prevalence graph."""
    graph = nx.from_numpy_array(prevalence)
    if graph.number_of_nodes() == 0:
        return {}
    return nx.spring_layout(graph, weight="weight", seed=42)


def _compute_community_layout(prevalence: np.ndarray) -> Dict[int, np.ndarray]:
    """Compute a modular/community layout from weighted prevalence graph."""
    graph = nx.from_numpy_array(prevalence)
    n_rois = prevalence.shape[0]
    if graph.number_of_edges() == 0:
        return _compute_circle_positions(n_rois)

    communities = list(nx.algorithms.community.greedy_modularity_communities(graph, weight="weight"))
    if len(communities) == 0:
        return _compute_circle_positions(n_rois)

    community_angles = np.linspace(0, 2 * np.pi, len(communities), endpoint=False)
    community_centers = {
        idx: np.array([1.8 * np.cos(angle), 1.8 * np.sin(angle)])
        for idx, angle in enumerate(community_angles)
    }

    pos = {}
    for community_idx, community_nodes in enumerate(communities):
        subgraph = graph.subgraph(sorted(community_nodes)).copy()
        if subgraph.number_of_nodes() == 1:
            node = next(iter(subgraph.nodes()))
            pos[node] = community_centers[community_idx]
            continue

        sub_pos = nx.spring_layout(subgraph, weight="weight", seed=42)
        sub_pos_array = np.array([sub_pos[node] for node in subgraph.nodes()])
        sub_pos_center = sub_pos_array.mean(axis=0)

        for node in subgraph.nodes():
            centered = np.array(sub_pos[node]) - sub_pos_center
            pos[node] = community_centers[community_idx] + 0.65 * centered

    # Ensure every node has a position
    for node_idx in range(n_rois):
        pos.setdefault(node_idx, np.array([0.0, 0.0]))

    return pos


def _get_layout_spec(prevalence: np.ndarray, layout_type: str) -> Dict:
    """Create plotting spec for a requested layout type."""
    n_rois = prevalence.shape[0]
    if layout_type == "circle":
        return {"positions": _compute_circle_positions(n_rois), "label_positions": None}
    if layout_type == "clustered_circle":
        order = _compute_clustered_circle_order(prevalence)
        return {"positions": _compute_circle_positions(n_rois, order=order), "label_positions": None}
    if layout_type == "force_fixed":
        positions = _compute_force_fixed_positions(prevalence)
        return {"positions": positions, "label_positions": None}
    if layout_type == "community":
        positions = _compute_community_layout(prevalence)
        return {"positions": positions, "label_positions": None}
    raise ValueError(f"Unsupported layout_type: {layout_type}")


def _plot_prevalence_network_with_layout(
    ax,
    prevalence: np.ndarray,
    roi_labels: List[str],
    title: str,
    layout_spec: Dict,
) -> None:
    """Plot prevalence network with externally supplied node layout."""
    positions = layout_spec["positions"]

    for i in range(prevalence.shape[0]):
        for j in range(i + 1, prevalence.shape[1]):
            w = float(prevalence[i, j])
            if w <= 0:
                continue
            ax.plot(
                [positions[i][0], positions[j][0]],
                [positions[i][1], positions[j][1]],
                color="black",
                alpha=min(0.95, max(0.08, w)),
                linewidth=0.4 + 2.8 * w,
                zorder=1,
            )

    xs = [positions[i][0] for i in range(prevalence.shape[0])]
    ys = [positions[i][1] for i in range(prevalence.shape[0])]
    ax.scatter(xs, ys, s=65, color="white", edgecolor="black", linewidth=1.0, zorder=3)

    for i, label in enumerate(roi_labels):
        x, y = positions[i]
        norm = max(np.sqrt(x**2 + y**2), 1e-6)
        ax.text(x + 0.12 * x / norm, y + 0.12 * y / norm, label, fontsize=5.5, ha="center", va="center")

    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.axis("off")
    ax.set_aspect("equal")


def create_edge_prevalence_network_figures(
    output_root: str,
    graph_dir: str,
    matrix_types: Optional[List[str]] = None,
    participants_group_column: Optional[str] = None,
    layout_types: Optional[List[str]] = None,
    dpi: int = 150,
) -> Dict[str, Path]:
    """
    Create threshold-wise edge-prevalence network plots.

    Produces one figure family per requested layout type:
    1) One figure per matrix type, single row with one panel per threshold
    2) One figure per matrix type with subgroup rows and threshold columns
    """
    output_root = Path(output_root)
    graph_dir = Path(graph_dir)

    network_df = _load_graph_table(graph_dir, level="network")
    network_json = graph_dir / "graphmetrics_desc-network.json"
    if not network_json.exists():
        raise FileNotFoundError(f"Missing graph metadata sidecar: {network_json}")

    with open(network_json, "r") as f:
        meta = json.load(f)

    threshold_mode = str(meta.get("ThresholdMode", "cost"))
    thresholds = [float(t) for t in meta.get("ThresholdValues", [])]
    positive_only = bool(meta.get("PositiveOnlyEdges", True))
    if not thresholds:
        raise RuntimeError("No threshold values found in graph metadata JSON.")

    if matrix_types is None:
        matrix_types = list(pd.unique(network_df["matrix_type"]))

    if layout_types is None:
        layout_types = list(EDGE_PREVALENCE_LAYOUT_TYPES)

    layout_types = [str(layout).strip().lower() for layout in layout_types]
    invalid_layouts = [layout for layout in layout_types if layout not in EDGE_PREVALENCE_LAYOUT_TYPES]
    if invalid_layouts:
        raise ValueError(
            f"Unsupported edge prevalence layout(s): {invalid_layouts}. "
            f"Supported values are: {EDGE_PREVALENCE_LAYOUT_TYPES}"
        )

    output_files: Dict[str, Path] = {}
    out_dir = graph_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    for matrix_type in matrix_types:
        mt_df = network_df[network_df["matrix_type"] == matrix_type].copy()
        subjects = sorted([str(s) for s in pd.unique(mt_df["subject_id"])])
        if len(subjects) == 0:
            continue

        # Determine node count from first available matrix
        sample_mat = None
        for sid in subjects:
            p = output_root / f"sub-{sid}" / f"corrmat_{matrix_type}.npy"
            if p.exists():
                sample_mat = np.load(p)
                break
        if sample_mat is None:
            continue

        n_rois = sample_mat.shape[0]
        roi_labels = _get_roi_labels_for_matrix(output_root, matrix_type, subjects, n_rois)

        mean_prevalence = _compute_mean_prevalence_matrix(
            output_root=output_root,
            subjects=subjects,
            matrix_type=matrix_type,
            threshold_mode=threshold_mode,
            thresholds=thresholds,
            positive_only=positive_only,
        )

        for layout_type in layout_types:
            layout_spec = _get_layout_spec(mean_prevalence, layout_type)

            # ------------------------------
            # Overall row (all subjects)
            # ------------------------------
            fig, axes = plt.subplots(1, len(thresholds), figsize=(4.8 * len(thresholds), 5.2), squeeze=False)

            for col_idx, thr in enumerate(thresholds):
                prevalence_sum = np.zeros((n_rois, n_rois), dtype=float)
                n_used = 0

                for sid in subjects:
                    mat_path = output_root / f"sub-{sid}" / f"corrmat_{matrix_type}.npy"
                    if not mat_path.exists():
                        continue
                    mat = np.load(mat_path)
                    if mat.shape != prevalence_sum.shape:
                        continue
                    adj = build_adjacency_from_corrmat(
                        matrix=mat,
                        threshold_mode=threshold_mode,
                        threshold_value=thr,
                        positive_only=positive_only,
                    )
                    prevalence_sum += adj
                    n_used += 1

                if n_used > 0:
                    prevalence = prevalence_sum / float(n_used)
                else:
                    prevalence = prevalence_sum

                ax = axes[0, col_idx]
                _plot_prevalence_network_with_layout(
                    ax=ax,
                    prevalence=prevalence,
                    roi_labels=roi_labels,
                    title=f"{matrix_type} | {threshold_mode}={thr:.2f}\nN={n_used}",
                    layout_spec=layout_spec,
                )

            fig.suptitle(
                f"Edge prevalence network: {matrix_type} ({layout_type})",
                fontsize=12,
                fontweight="bold",
            )
            plt.tight_layout()
            overall_path = out_dir / f"edgeprevalence_{matrix_type}_{layout_type}_by-threshold.png"
            fig.savefig(overall_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            output_files[f"edgeprevalence_{matrix_type}_{layout_type}_overall"] = overall_path

            # ------------------------------
            # Subgroup rows
            # ------------------------------
            if participants_group_column and participants_group_column in mt_df.columns:
                group_df = mt_df.dropna(subset=[participants_group_column]).copy()
                if not group_df.empty:
                    groups = sorted([str(g) for g in pd.unique(group_df[participants_group_column])])
                    fig, axes = plt.subplots(
                        len(groups),
                        len(thresholds),
                        figsize=(4.8 * len(thresholds), 4.6 * len(groups)),
                        squeeze=False,
                    )

                    for row_idx, group_value in enumerate(groups):
                        grp_subjects = sorted(
                            [
                                str(s)
                                for s in pd.unique(
                                    group_df.loc[group_df[participants_group_column].astype(str) == group_value, "subject_id"]
                                )
                            ]
                        )

                        for col_idx, thr in enumerate(thresholds):
                            prevalence_sum = np.zeros((n_rois, n_rois), dtype=float)
                            n_used = 0

                            for sid in grp_subjects:
                                mat_path = output_root / f"sub-{sid}" / f"corrmat_{matrix_type}.npy"
                                if not mat_path.exists():
                                    continue
                                mat = np.load(mat_path)
                                if mat.shape != prevalence_sum.shape:
                                    continue
                                adj = build_adjacency_from_corrmat(
                                    matrix=mat,
                                    threshold_mode=threshold_mode,
                                    threshold_value=thr,
                                    positive_only=positive_only,
                                )
                                prevalence_sum += adj
                                n_used += 1

                            if n_used > 0:
                                prevalence = prevalence_sum / float(n_used)
                            else:
                                prevalence = prevalence_sum

                            ax = axes[row_idx, col_idx]
                            _plot_prevalence_network_with_layout(
                                ax=ax,
                                prevalence=prevalence,
                                roi_labels=roi_labels,
                                title=f"{group_value} | {threshold_mode}={thr:.2f}\nN={n_used}",
                                layout_spec=layout_spec,
                            )

                    fig.suptitle(
                        f"Edge prevalence by subgroup: {matrix_type} ({participants_group_column}, {layout_type})",
                        fontsize=12,
                        fontweight="bold",
                    )
                    plt.tight_layout()
                    subgroup_path = out_dir / f"edgeprevalence_{matrix_type}_{layout_type}_by-group-threshold.png"
                    fig.savefig(subgroup_path, dpi=dpi, bbox_inches="tight")
                    plt.close(fig)
                    output_files[f"edgeprevalence_{matrix_type}_{layout_type}_group"] = subgroup_path

    return output_files


def create_graph_metric_summary_figures(
    graph_dir: str,
    participants_group_column: Optional[str] = None,
    matrix_types: Optional[List[str]] = None,
    matrix_display_names: Optional[Dict[str, str]] = None,
    dpi: int = 150,
) -> Dict[str, Path]:
    """
    Create summary dot+box figures for network and node graph metrics.

    For each metric, creates:
    1) intervention/control comparison
    2) subgroup-stratified comparison (if group column is provided)

    Returns
    -------
    output_files : dict
        Mapping of figure keys to output paths.
    """
    graph_dir = Path(graph_dir)
    if not graph_dir.exists():
        raise FileNotFoundError(f"Graph directory not found: {graph_dir}")

    network_df = _load_graph_table(graph_dir, level="network")
    node_df = _load_graph_table(graph_dir, level="node")

    if matrix_types is None:
        matrix_types = ["intervention", "control"]

    if matrix_display_names is None:
        matrix_display_names = {
            "intervention": "intervention",
            "control": "control",
        }

    output_dir = graph_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_files: Dict[str, Path] = {}

    # Keep only requested matrix types when available
    if "matrix_type" in network_df.columns:
        network_df = network_df[network_df["matrix_type"].isin(matrix_types)].copy()
    if "matrix_type" in node_df.columns:
        node_df = node_df[node_df["matrix_type"].isin(matrix_types)].copy()

    if network_df.empty or node_df.empty:
        raise RuntimeError("Graph tables are empty after filtering by matrix types.")

    # Display condition labels
    network_df["condition_label"] = network_df["matrix_type"].map(matrix_display_names).fillna(network_df["matrix_type"])
    node_df["condition_label"] = node_df["matrix_type"].map(matrix_display_names).fillna(node_df["matrix_type"])
    condition_order = [matrix_display_names.get(m, m) for m in matrix_types]

    # ------------------------------------------------------------------
    # Network-level figures
    # ------------------------------------------------------------------
    for metric in GRAPH_METRIC_COLUMNS:
        if metric not in network_df.columns:
            continue

        # Condition figure
        cond_fig = output_dir / f"network_{metric}_by-condition.png"
        _dot_box_plot(
            df=network_df,
            x_col="condition_label",
            y_col=metric,
            title=f"Network {metric}: intervention vs control",
            output_path=cond_fig,
            order=condition_order,
            participant_col="participant_id" if "participant_id" in network_df.columns else None,
            condition_col="matrix_type" if "matrix_type" in network_df.columns else None,
            condition_order=matrix_types,
            condition_display_names=matrix_display_names,
            dpi=dpi,
        )
        output_files[f"network_{metric}_condition"] = cond_fig

        # Group-stratified figure
        if participants_group_column and participants_group_column in network_df.columns:
            strat_df = network_df.dropna(subset=[participants_group_column]).copy()
            if not strat_df.empty:
                group_values = sorted([str(g) for g in pd.unique(strat_df[participants_group_column])])
                strat_df["group_condition"] = (
                    strat_df[participants_group_column].astype(str)
                    + " "
                    + strat_df["condition_label"].astype(str)
                )
                order = []
                for g in group_values:
                    for c in condition_order:
                        order.append(f"{g} {c}")

                strat_fig = output_dir / f"network_{metric}_by-group-condition.png"
                _dot_box_plot(
                    df=strat_df,
                    x_col="group_condition",
                    y_col=metric,
                    title=f"Network {metric}: subgroup-stratified",
                    output_path=strat_fig,
                    order=order,
                    participant_col="participant_id" if "participant_id" in strat_df.columns else None,
                    condition_col="matrix_type" if "matrix_type" in strat_df.columns else None,
                    condition_order=matrix_types,
                    condition_display_names=matrix_display_names,
                    group_col=participants_group_column,
                    figsize=(11, 6),
                    dpi=dpi,
                )
                output_files[f"network_{metric}_group_condition"] = strat_fig

    # ------------------------------------------------------------------
    # Node-level figures (one set per ROI)
    # ------------------------------------------------------------------
    if "roi" not in node_df.columns:
        raise RuntimeError("Node table must include a 'roi' column.")

    for roi in sorted(pd.unique(node_df["roi"])):
        roi_df = node_df[node_df["roi"] == roi].copy()
        if roi_df.empty:
            continue

        roi_safe = "".join(ch if (ch.isalnum() or ch in {"_", "-"}) else "-" for ch in str(roi))

        for metric in GRAPH_METRIC_COLUMNS:
            if metric not in roi_df.columns:
                continue

            cond_fig = output_dir / f"node_{roi_safe}_{metric}_by-condition.png"
            _dot_box_plot(
                df=roi_df,
                x_col="condition_label",
                y_col=metric,
                title=f"Node {metric} ({roi}): intervention vs control",
                output_path=cond_fig,
                order=condition_order,
                participant_col="participant_id" if "participant_id" in roi_df.columns else None,
                condition_col="matrix_type" if "matrix_type" in roi_df.columns else None,
                condition_order=matrix_types,
                condition_display_names=matrix_display_names,
                dpi=dpi,
            )
            output_files[f"node_{roi_safe}_{metric}_condition"] = cond_fig

            if participants_group_column and participants_group_column in roi_df.columns:
                strat_df = roi_df.dropna(subset=[participants_group_column]).copy()
                if not strat_df.empty:
                    group_values = sorted([str(g) for g in pd.unique(strat_df[participants_group_column])])
                    strat_df["group_condition"] = (
                        strat_df[participants_group_column].astype(str)
                        + " "
                        + strat_df["condition_label"].astype(str)
                    )
                    order = []
                    for g in group_values:
                        for c in condition_order:
                            order.append(f"{g} {c}")

                    strat_fig = output_dir / f"node_{roi_safe}_{metric}_by-group-condition.png"
                    _dot_box_plot(
                        df=strat_df,
                        x_col="group_condition",
                        y_col=metric,
                        title=f"Node {metric} ({roi}): subgroup-stratified",
                        output_path=strat_fig,
                        order=order,
                        participant_col="participant_id" if "participant_id" in strat_df.columns else None,
                        condition_col="matrix_type" if "matrix_type" in strat_df.columns else None,
                        condition_order=matrix_types,
                        condition_display_names=matrix_display_names,
                        group_col=participants_group_column,
                        figsize=(11, 6),
                        dpi=dpi,
                    )
                    output_files[f"node_{roi_safe}_{metric}_group_condition"] = strat_fig

    return output_files
