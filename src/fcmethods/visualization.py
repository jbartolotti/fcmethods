"""
Visualization utilities for correlation matrices and functional connectivity.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict, List
import warnings


def _normalize_roi_label(label: str) -> str:
    """Normalize ROI label for robust cluster matching."""
    label_norm = label.strip().lower()
    for prefix in ("l_", "r_", "lh_", "rh_", "left_", "right_"):
        if label_norm.startswith(prefix):
            return label_norm[len(prefix):]
    return label_norm


def _get_cluster_boundaries(
    roi_labels: Optional[List[str]],
    roi_clusters: Optional[Dict[str, List[str]]],
) -> List[float]:
    """Return boundary positions (between cells) where cluster assignment changes."""
    if roi_labels is None or roi_clusters is None:
        return []

    # Build reverse lookup from ROI label -> cluster name
    normalized_cluster_lookup = {}
    for cluster_name, cluster_rois in roi_clusters.items():
        for roi in cluster_rois:
            normalized_cluster_lookup[_normalize_roi_label(roi)] = cluster_name

    cluster_assignments = []
    for roi_label in roi_labels:
        cluster_assignments.append(
            normalized_cluster_lookup.get(_normalize_roi_label(roi_label), "__unclustered__")
        )

    boundaries = []
    for idx in range(len(cluster_assignments) - 1):
        if cluster_assignments[idx] != cluster_assignments[idx + 1]:
            boundaries.append(idx + 0.5)

    return boundaries


def remove_diagonal(matrix: np.ndarray, set_to_nan: bool = True) -> np.ndarray:
    """
    Remove or mask the diagonal of a matrix.
    
    Parameters
    ----------
    matrix : np.ndarray
        Input matrix (typically 2D)
    set_to_nan : bool, optional
        If True, set diagonal to NaN (better for color scale in heatmaps).
        If False, set diagonal to 0. Default: True
    
    Returns
    -------
    matrix_no_diag : np.ndarray
        Copy of matrix with diagonal removed
    """
    matrix_copy = matrix.copy()
    np.fill_diagonal(matrix_copy, np.nan if set_to_nan else 0)
    return matrix_copy


def plot_correlation_matrices(
    matrices: Dict[str, np.ndarray],
    roi_labels: Optional[List[str]] = None,
    roi_clusters: Optional[Dict[str, List[str]]] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "RdBu_r",
    figsize: tuple = (15, 5),
    title_prefix: str = "",
) -> plt.Figure:
    """
    Plot multiple correlation matrices in a single figure.
    
    Parameters
    ----------
    matrices : dict
        Dictionary mapping matrix names (e.g., "intervention", "control", "diff") 
        to correlation matrices (n_rois x n_rois)
    roi_labels : list, optional
        Labels for ROIs (rows/columns). If None, uses numeric indices
    roi_clusters : dict, optional
        ROI cluster definitions as {cluster_name: [roi1, roi2, ...]}.
        Cluster boundaries are drawn as black lines where adjacent ROIs belong
        to different clusters. Matching is hemisphere-aware: labels like L_X
        and R_X both match cluster ROI "X".
    vmin, vmax : float, optional
        Color scale limits. If None, computed from data
    cmap : str, optional
        Colormap name (default: "RdBu_r")
    figsize : tuple, optional
        Figure size (width, height). Default: (15, 5)
    title_prefix : str, optional
        Prefix for subplot titles (e.g., "sub-2002")
    
    Returns
    -------
    fig : plt.Figure
        The matplotlib figure object
    """
    
    n_matrices = len(matrices)
    fig, axes = plt.subplots(1, n_matrices, figsize=figsize)
    
    # Handle single matrix case
    if n_matrices == 1:
        axes = [axes]
    
    # Compute color scale from all matrices if not provided
    if vmin is None or vmax is None:
        all_data = np.concatenate([remove_diagonal(m).flatten() for m in matrices.values()])
        all_data = all_data[~np.isnan(all_data)]
        if vmin is None:
            vmin = np.percentile(all_data, 1)  # Use 1st percentile to avoid outliers
        if vmax is None:
            vmax = np.percentile(all_data, 99)  # Use 99th percentile
    
    # Plot each matrix
    cluster_boundaries = _get_cluster_boundaries(roi_labels, roi_clusters)
    for ax, (matrix_name, matrix) in zip(axes, matrices.items()):
        # Remove diagonal
        matrix_viz = remove_diagonal(matrix, set_to_nan=True)
        
        # Plot heatmap
        im = ax.imshow(matrix_viz, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
        
        # Set labels
        if roi_labels is not None:
            ax.set_xticks(range(len(roi_labels)))
            ax.set_yticks(range(len(roi_labels)))
            ax.set_xticklabels(roi_labels, rotation=45, ha='right', fontsize=8)
            ax.set_yticklabels(roi_labels, fontsize=8)

        # Draw cluster boundary lines
        for boundary in cluster_boundaries:
            ax.axhline(boundary, color='black', linewidth=1.8, alpha=0.9)
            ax.axvline(boundary, color='black', linewidth=1.8, alpha=0.9)
        
        ax.set_title(f"{title_prefix}{matrix_name}" if title_prefix else matrix_name, 
                     fontsize=12, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Correlation (z)', fontsize=10)
    
    plt.tight_layout()
    return fig


def visualize_subject_corrmat(
    subject_id: str,
    output_dir: Path,
    output_root: Path,
    roi_labels: Optional[List[str]] = None,
    roi_clusters: Optional[Dict[str, List[str]]] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "RdBu_r",
    figsize: tuple = (15, 5),
    dpi: int = 150,
) -> Optional[Path]:
    """
    Create and save a heatmap figure for a single subject's correlation matrices.
    
    Loads intervention, control, and difference matrices and creates a 3-panel figure.
    
    Parameters
    ----------
    subject_id : str
        Subject ID (e.g., "2002")
    output_dir : Path
        Directory containing corrmat_*.npy files for this subject
    output_root : Path
        Root output directory for saving figures (will create figures/ subdirectory)
    roi_labels : list, optional
        Labels for ROIs. If None, uses numeric indices
    roi_clusters : dict, optional
        ROI cluster definitions for drawing boundary lines.
    vmin, vmax : float, optional
        Color scale limits. If None, computed from data
    cmap : str, optional
        Colormap name. Default: "RdBu_r"
    figsize : tuple, optional
        Figure size. Default: (15, 5)
    dpi : int, optional
        DPI for saved figure. Default: 150
    
    Returns
    -------
    fig_path : Path or None
        Path to saved figure, or None if matrices not found
    """
    
    # Try to load matrices
    matrices = {}
    for matrix_type in ["intervention", "control", "diff"]:
        corrmat_file = output_dir / f"corrmat_{matrix_type}.npy"
        if corrmat_file.exists():
            matrices[matrix_type] = np.load(corrmat_file)
    
    if not matrices:
        warnings.warn(f"No correlation matrices found for sub-{subject_id} in {output_dir}")
        return None
    
    # Create figure
    fig = plot_correlation_matrices(
        matrices,
        roi_labels=roi_labels,
        roi_clusters=roi_clusters,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        figsize=figsize,
        title_prefix=f"sub-{subject_id} ",
    )
    
    # Save figure
    figures_dir = output_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig_path = figures_dir / f"sub-{subject_id}_corrmat_heatmaps.png"
    fig.savefig(fig_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    return fig_path


def visualize_group_corrmat(
    output_root: Path,
    subjects: Optional[List[str]] = None,
    roi_labels: Optional[List[str]] = None,
    roi_clusters: Optional[Dict[str, List[str]]] = None,
    cmap: str = "RdBu_r",
    figsize: tuple = (15, 5),
    dpi: int = 150,
) -> Optional[Path]:
    """
    Create and save a group-averaged heatmap figure.
    
    Loads and averages correlation matrices across subjects, creating
    a 3-panel figure (intervention, control, difference).
    
    Parameters
    ----------
    output_root : Path
        Root output directory containing sub-*/corrmat_*.npy files
    subjects : list, optional
        List of subject IDs to include. If None, includes all with corrmat files
    roi_labels : list, optional
        Labels for ROIs. If None, uses numeric indices
    roi_clusters : dict, optional
        ROI cluster definitions for drawing boundary lines.
    cmap : str, optional
        Colormap name. Default: "RdBu_r"
    figsize : tuple, optional
        Figure size. Default: (15, 5)
    dpi : int, optional
        DPI for saved figure. Default: 150
    
    Returns
    -------
    fig_path : Path or None
        Path to saved figure, or None if no matrices found
    """
    
    output_root = Path(output_root)
    
    # Find all subjects if not specified
    if subjects is None:
        subject_dirs = sorted(output_root.glob("sub-*"))
        subjects = [d.name.replace("sub-", "") for d in subject_dirs]
    
    # Load all matrices
    all_matrices = {"intervention": [], "control": [], "diff": []}
    
    for sub_id in subjects:
        sub_dir = output_root / f"sub-{sub_id}"
        
        for matrix_type in ["intervention", "control", "diff"]:
            corrmat_file = sub_dir / f"corrmat_{matrix_type}.npy"
            if corrmat_file.exists():
                matrix = np.load(corrmat_file)
                all_matrices[matrix_type].append(matrix)
    
    # Check if we found any matrices
    if not any(all_matrices.values()):
        warnings.warn(f"No correlation matrices found in {output_root}")
        return None
    
    # Average matrices (handling different numbers of available matrices)
    avg_matrices = {}
    for matrix_type, matrices_list in all_matrices.items():
        if matrices_list:
            # Stack and average
            stacked = np.stack(matrices_list, axis=0)
            avg_matrices[matrix_type] = np.nanmean(stacked, axis=0)
    
    # Create figure
    fig = plot_correlation_matrices(
        avg_matrices,
        roi_labels=roi_labels,
        roi_clusters=roi_clusters,
        cmap=cmap,
        figsize=figsize,
        title_prefix="Group Average ",
    )
    
    # Save figure
    figures_dir = output_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig_path = figures_dir / "group_corrmat_heatmaps.png"
    fig.savefig(fig_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    return fig_path
