"""
Mixed-effects modeling utilities for network-level graph metrics.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.formula.api import mixedlm
from statsmodels.stats.multitest import multipletests


NETWORK_METRIC_COLUMNS = [
    "degree",
    "cost",
    "avg_path_distance",
    "clustering_coefficient",
    "global_efficiency",
    "local_efficiency",
    "betweenness_centrality",
    "modularity",
]


def _load_network_metrics_for_modeling(graph_dir: Path) -> Tuple[pd.DataFrame, str]:
    """Load preferred network metrics table: AUCnorm > AUC > raw."""
    graph_dir = Path(graph_dir)

    candidates = [
        (graph_dir / "graphmetrics_desc-networkAUCnorm.tsv", "AUCnorm"),
        (graph_dir / "graphmetrics_desc-networkAUC.tsv", "AUC"),
        (graph_dir / "graphmetrics_desc-network.tsv", "raw"),
    ]

    for path, label in candidates:
        if path.exists():
            df = pd.read_csv(path, sep="\t")
            return df, label

    raise FileNotFoundError(f"Could not find network metrics TSV in {graph_dir}")


def _prepare_paired_condition_dataframe(
    df: pd.DataFrame,
    condition_column: str,
    intervention_label: str,
    control_label: str,
    subject_column: str,
) -> pd.DataFrame:
    """Filter to subjects with both conditions and add centered condition contrast."""
    required = [subject_column, condition_column]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df = df[df[condition_column].isin([intervention_label, control_label])].copy()

    pair_counts = (
        df[[subject_column, condition_column]]
        .drop_duplicates()
        .groupby(subject_column)[condition_column]
        .nunique()
    )
    paired_subjects = pair_counts[pair_counts == 2].index.tolist()
    df = df[df[subject_column].isin(paired_subjects)].copy()

    contrast_map = {intervention_label: 0.5, control_label: -0.5}
    df["condition_contrast"] = df[condition_column].map(contrast_map).astype(float)

    return df


def _fit_one_metric_mixedlm(
    df: pd.DataFrame,
    metric: str,
    subject_column: str,
) -> Optional[Dict[str, float]]:
    """Fit mixed model for one metric and extract condition effect statistics."""
    sub = df[[subject_column, "condition_contrast", metric]].dropna().copy()
    if sub.empty:
        return None

    # Need repeated observations per subject for random-intercept mixed model
    subject_counts = sub.groupby(subject_column).size()
    valid_subjects = subject_counts[subject_counts >= 2].index.tolist()
    sub = sub[sub[subject_column].isin(valid_subjects)]
    if sub.empty:
        return None

    # Build paired differences for an intuitive effect size (Cohen's dz)
    pivot = sub.pivot_table(index=subject_column, columns="condition_contrast", values=metric, aggfunc="mean")
    if (-0.5 not in pivot.columns) or (0.5 not in pivot.columns):
        return None

    diffs = pivot[0.5] - pivot[-0.5]
    diffs = diffs.dropna()
    if len(diffs) < 2:
        return None

    try:
        model = mixedlm(f"{metric} ~ condition_contrast", data=sub, groups=sub[subject_column])
        result = model.fit(reml=False)
    except Exception:
        return None

    beta = float(result.params.get("condition_contrast", np.nan))
    se = float(result.bse.get("condition_contrast", np.nan))
    z_value = float(result.tvalues.get("condition_contrast", np.nan))
    p_value = float(result.pvalues.get("condition_contrast", np.nan))

    ci = result.conf_int()
    if "condition_contrast" in ci.index:
        ci_low = float(ci.loc["condition_contrast", 0])
        ci_high = float(ci.loc["condition_contrast", 1])
    else:
        ci_low, ci_high = np.nan, np.nan

    sd_y = float(sub[metric].std(ddof=1)) if len(sub) > 1 else np.nan
    standardized_beta = beta / sd_y if (sd_y is not None and sd_y > 0) else np.nan

    diff_mean = float(diffs.mean())
    diff_sd = float(diffs.std(ddof=1)) if len(diffs) > 1 else np.nan
    cohen_dz = diff_mean / diff_sd if (diff_sd is not None and diff_sd > 0) else np.nan

    return {
        "metric": metric,
        "n_subjects": int(len(diffs)),
        "n_observations": int(len(sub)),
        "beta_condition_contrast": beta,
        "se_condition_contrast": se,
        "z_condition_contrast": z_value,
        "p_condition_contrast": p_value,
        "ci95_low_condition_contrast": ci_low,
        "ci95_high_condition_contrast": ci_high,
        "standardized_beta_condition_contrast": standardized_beta,
        "cohen_dz": cohen_dz,
        "aic": float(result.aic),
        "bic": float(result.bic),
        "llf": float(result.llf),
    }


def _plot_ranked_effect_size(results_df: pd.DataFrame, output_path: Path, dpi: int = 150) -> None:
    """Create ranked effect-size figure."""
    df = results_df.copy()
    df = df.dropna(subset=["cohen_dz"]).sort_values("cohen_dz", ascending=False)
    if df.empty:
        return

    colors = ["#4C78A8" if x >= 0 else "#E45756" for x in df["cohen_dz"]]

    fig, ax = plt.subplots(figsize=(8, max(4, 0.45 * len(df))))
    y_pos = np.arange(len(df))
    ax.barh(y_pos, df["cohen_dz"], color=colors, alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["metric"])
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=1.0)
    ax.set_xlabel("Effect size (Cohen's dz)")
    ax.set_title("Network metrics ranked by condition effect size", fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_ranked_significance(results_df: pd.DataFrame, output_path: Path, dpi: int = 150) -> None:
    """Create ranked significance figure (-log10 p)."""
    df = results_df.copy()
    df = df.dropna(subset=["p_condition_contrast"]).copy()
    if df.empty:
        return

    eps = 1e-300
    df["neg_log10_p"] = -np.log10(np.clip(df["p_condition_contrast"].astype(float), eps, 1.0))
    df = df.sort_values("neg_log10_p", ascending=False)

    fig, ax = plt.subplots(figsize=(8, max(4, 0.45 * len(df))))
    y_pos = np.arange(len(df))
    ax.barh(y_pos, df["neg_log10_p"], color="#59A14F", alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["metric"])
    ax.invert_yaxis()
    ax.set_xlabel("-log10(p)")
    ax.set_title("Network metrics ranked by significance", fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.35)

    # p=0.05 line
    ax.axvline(-np.log10(0.05), color="black", linestyle=":", linewidth=1.0)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def run_network_metric_mixed_models(
    output_root: str,
    condition_column: str = "matrix_type",
    intervention_label: str = "intervention",
    control_label: str = "control",
    subject_column: str = "subject_id",
    metrics: Optional[List[str]] = None,
    dpi: int = 150,
) -> Dict[str, Path]:
    """
    Run mixed-effects models for each network metric using centered condition contrast.

    Model per metric:
        metric ~ condition_contrast + (1 | subject)

    with condition_contrast coded as +0.5 (intervention), -0.5 (control).
    """
    output_root = Path(output_root)
    graph_dir = output_root / "graph"
    stats_dir = graph_dir / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    network_df, source_table = _load_network_metrics_for_modeling(graph_dir)

    model_df = _prepare_paired_condition_dataframe(
        df=network_df,
        condition_column=condition_column,
        intervention_label=intervention_label,
        control_label=control_label,
        subject_column=subject_column,
    )

    if metrics is None:
        metrics = [m for m in NETWORK_METRIC_COLUMNS if m in model_df.columns]

    rows = []
    for metric in metrics:
        fit_row = _fit_one_metric_mixedlm(model_df, metric=metric, subject_column=subject_column)
        if fit_row is not None:
            rows.append(fit_row)

    if not rows:
        raise RuntimeError("No mixed models could be fit. Check data availability and metric columns.")

    results_df = pd.DataFrame(rows)

    # FDR correction across metrics
    pvals = results_df["p_condition_contrast"].values.astype(float)
    _, p_fdr, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
    results_df["p_fdr_bh"] = p_fdr

    results_df = results_df.sort_values("p_condition_contrast", ascending=True).reset_index(drop=True)

    results_tsv = stats_dir / "mixedlm_desc-networkmetrics_conditioncontrast.tsv"
    results_json = stats_dir / "mixedlm_desc-networkmetrics_conditioncontrast.json"
    results_df.to_csv(results_tsv, sep="\t", index=False)

    metadata = {
        "Description": "Mixed-effects models for network-level graph metrics",
        "Model": "metric ~ condition_contrast + (1|subject)",
        "ConditionContrastCoding": {
            intervention_label: 0.5,
            control_label: -0.5,
        },
        "SourceTable": source_table,
        "Metrics": metrics,
        "MultipleComparisonCorrection": "Benjamini-Hochberg FDR across metrics",
    }
    with open(results_json, "w") as f:
        json.dump(metadata, f, indent=2)

    effect_fig = stats_dir / "mixedlm_ranked_effectsize_cohendz.png"
    sig_fig = stats_dir / "mixedlm_ranked_significance_neglog10p.png"
    _plot_ranked_effect_size(results_df, effect_fig, dpi=dpi)
    _plot_ranked_significance(results_df, sig_fig, dpi=dpi)

    return {
        "results_tsv": results_tsv,
        "results_json": results_json,
        "ranked_effectsize_figure": effect_fig,
        "ranked_significance_figure": sig_fig,
    }
