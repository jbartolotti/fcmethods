"""
Visualization utilities for graph metrics outputs.
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


GRAPH_METRIC_COLUMNS = [
    "degree",
    "cost",
    "avg_path_distance",
    "clustering_coefficient",
    "global_efficiency",
    "local_efficiency",
    "betweenness_centrality",
]


def _load_graph_table(graph_dir: Path, level: str) -> pd.DataFrame:
    """Load AUC table when available, otherwise load threshold-wise table."""
    graph_dir = Path(graph_dir)

    auc_path = graph_dir / f"graphmetrics_desc-{level}AUC.tsv"
    raw_path = graph_dir / f"graphmetrics_desc-{level}.tsv"

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
    figsize: tuple = (8, 6),
    dpi: int = 150,
) -> None:
    """Create a dot+box plot and save figure."""
    if df.empty:
        return

    plot_df = df[[x_col, y_col]].dropna().copy()
    if plot_df.empty:
        return

    if order is None:
        categories = list(pd.unique(plot_df[x_col]))
    else:
        categories = [cat for cat in order if cat in set(plot_df[x_col])]

    if not categories:
        return

    y_data = [plot_df.loc[plot_df[x_col] == cat, y_col].values for cat in categories]

    fig, ax = plt.subplots(figsize=figsize)

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
                        figsize=(11, 6),
                        dpi=dpi,
                    )
                    output_files[f"node_{roi_safe}_{metric}_group_condition"] = strat_fig

    return output_files
