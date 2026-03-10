"""
User-facing interface for timecourse export and network analysis with logging and reporting.

This module provides high-level functions with comprehensive console output and
error handling for both timecourse export and correlation matrix computation.
"""

import json
import csv
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime

import numpy as np
import pandas as pd

from .timecourse_io import export_timecourses_to_bids
from .network_analysis import (
    get_bids_files,
    compute_subject_correlation_matrices,
)
from .visualization import (
    visualize_subject_corrmat,
    visualize_group_corrmat,
)
from .graph_analysis import (
    build_adjacency_from_corrmat,
    compute_node_graph_metrics,
    compute_network_graph_metrics,
    compute_auc_by_group,
    save_graph_outputs,
)


def _infer_roi_labels_from_corrmat_json(
    output_root: Path,
    subjects: List[str],
) -> Optional[List[str]]:
    """Infer ROI labels from a subject corrmat/cormat JSON sidecar."""
    candidate_filenames = [
        "corrmat_control.json",
        "corrmat_intervention.json",
        "corrmat_diff.json",
        "cormat_control.json",
        "cormat_intervention.json",
        "cormat_diff.json",
    ]

    for sub_id in subjects:
        sub_dir = output_root / f"sub-{sub_id}"
        if not sub_dir.exists():
            continue

        # Try expected names first
        for filename in candidate_filenames:
            json_path = sub_dir / filename
            if not json_path.exists():
                continue

            try:
                with open(json_path, "r") as f:
                    metadata = json.load(f)
                rois = metadata.get("ROIs")
            except Exception:
                continue

            if isinstance(rois, list) and rois:
                return [str(roi) for roi in rois]
            if isinstance(rois, dict) and rois:
                return [str(roi) for roi in rois.keys()]

        # Fallback: any matching corrmat/cormat JSON in subject directory
        for json_path in sorted(sub_dir.glob("*cor*mat*.json")):
            try:
                with open(json_path, "r") as f:
                    metadata = json.load(f)
                rois = metadata.get("ROIs")
            except Exception:
                continue

            if isinstance(rois, list) and rois:
                return [str(roi) for roi in rois]
            if isinstance(rois, dict) and rois:
                return [str(roi) for roi in rois.keys()]

    return None


def _normalize_subject_id(subject_id: str) -> str:
    """Normalize subject IDs to bare numeric/string identifiers without sub- prefix."""
    subject_id = str(subject_id).strip()
    if subject_id.startswith("sub-"):
        return subject_id.replace("sub-", "", 1)
    return subject_id


def _load_participants_rows(output_root: Path) -> Optional[List[Dict[str, str]]]:
    """Load participants.tsv from the base BIDS directory if available."""
    participants_path = output_root.parent.parent / "participants.tsv"
    if not participants_path.exists():
        return None

    with open(participants_path, "r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader)


def _is_include_value(value: Optional[str]) -> bool:
    """Interpret common truthy include values from participants.tsv."""
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _get_default_subjects_and_groups(
    output_root: Path,
    available_subjects: List[str],
    participants_group_column: Optional[str] = None,
    participants_include_column: str = "include",
) -> tuple:
    """Resolve default subjects and optional groupings from participants.tsv."""
    participants_rows = _load_participants_rows(output_root)
    available_subjects_set = set(available_subjects)
    group_assignments = {}

    if not participants_rows:
        return available_subjects, group_assignments, False, False

    has_include_column = any(participants_include_column in row for row in participants_rows)
    selected_subjects = []

    for row in participants_rows:
        participant_id = row.get("participant_id")
        if not participant_id:
            continue

        subject_id = _normalize_subject_id(participant_id)
        if subject_id not in available_subjects_set:
            continue

        if has_include_column and not _is_include_value(row.get(participants_include_column)):
            continue

        selected_subjects.append(subject_id)

        if participants_group_column and participants_group_column in row:
            group_value = str(row.get(participants_group_column, "")).strip()
            if group_value:
                group_assignments[subject_id] = group_value

    if not selected_subjects:
        return available_subjects, group_assignments, has_include_column, participants_group_column is not None

    return selected_subjects, group_assignments, has_include_column, participants_group_column is not None


def _sanitize_group_value(value: str) -> str:
    """Convert group values to safe filename fragments."""
    sanitized = []
    for char in str(value):
        if char.isalnum() or char in {"-", "_"}:
            sanitized.append(char)
        elif char.isspace():
            sanitized.append("-")
    return "".join(sanitized).strip("-") or "group"


def export_timecourses_to_bids_with_reporting(
    csv_path: str,
    bids_root: str,
    network_label: str,
    preamble_cols: Optional[List[str]] = None,
    filename_prefix: str = "stat-mean_timeseries",
    file_format: str = "tsv",
    create_derivatives_subdir: bool = True,
    dry_run: bool = False,
    repetition_time: Optional[float] = None,
    roi_metadata: Optional[Dict[str, Dict]] = None,
    task_label: Optional[str] = None,
    condition_to_task_mapping: Optional[Dict[str, str]] = None,
    processing_description: Optional[str] = None,
    output_subdir: str = "func",
    censor_convention: str = "standard",
    verbose: bool = True,
) -> Dict[str, List[str]]:
    """
    Export timecourse data to BIDS with comprehensive logging and reporting.
    
    This function wraps export_timecourses_to_bids and provides detailed
    console output about the export process, validation, and results.
    
    Parameters
    ----------
    csv_path : str
        Path to the input CSV file
    bids_root : str
        Root path of the BIDS dataset
    network_label : str
        Network/parcellation label (e.g., "default", "schaefer400")
        Used as subdirectory in derivatives
    preamble_cols : list, optional
        Column names for metadata. Default: ["slicenum", "condition", "subnum", "run", "time", "censor", "subgroup"]
    filename_prefix : str, optional
        Descriptor for output files (e.g., "stat-mean_timeseries", "confounds"). 
        Default: "stat-mean_timeseries"
    file_format : str, optional
        Output format: "tsv", "csv", or "npy". Default: "tsv"
    create_derivatives_subdir : bool, optional
        If True, organize by derivatives/{network_label}/sub-{subnum}
        If False, organize by derivatives/sub-{subnum}. Default: True
    dry_run : bool, optional
        If True, preview the output without writing files. Default: False
    repetition_time : float, optional
        TR in seconds (e.g., 2.0 for 2s TR). Will be saved in JSON sidecar.
    roi_metadata : dict, optional
        Dictionary mapping ROI names to metadata dicts. Example:
        {"L_lPFC": {"hemisphere": "left", "radius_mm": 6, "center_xyz": [x, y, z]}}
    task_label : str, optional
        Task label for BIDS filename (e.g., "rest", "nback")
    condition_to_task_mapping : dict, optional
        Dictionary mapping condition values to task labels. Example:
        {"A": "rest-drug", "B": "rest-placebo"}
        If provided, will override task_label based on condition column values.
    processing_description : str, optional
        Description of processing steps applied to timecourses
    output_subdir : str, optional
        Subdirectory for output files (e.g., "func", "timeseries"). Default: "func"
    censor_convention : str, optional
        Convention used for the censor column. Options:
        - "standard": 1=censored/excluded, 0=retained (BIDS default)
        - "inverted": 1=retained, 0=censored/excluded
        Default: "standard"
    output_subdir : str, optional
        Subdirectory for output files (e.g., "func", "timeseries"). Default: "func"
    verbose : bool, optional
        If True, print detailed status messages and summaries. Default: True
    
    Returns
    -------
    output_files : dict
        Dictionary mapping subject IDs to lists of output file paths (TSV and JSON)
    
    Raises
    ------
    FileNotFoundError
        If the input CSV file does not exist
    
    Examples
    --------
    >>> output_files = export_timecourses_to_bids_with_reporting(
    ...     csv_path="timecourses.csv",
    ...     bids_root="/path/to/BIDS",
    ...     network_label="default",
    ...     repetition_time=2.0,
    ...     condition_to_task_mapping={"A": "rest-drug", "B": "rest-placebo"},
    ...     dry_run=True
    ... )
    """
    
    if verbose:
        print("=" * 80)
        print("Timecourse Export to BIDS")
        print("=" * 80)
        
        if dry_run:
            print("\n[DRY-RUN MODE] Files will NOT be written. Run with dry_run=False to export.\n")
        else:
            print("\n[WRITING FILES] Exporting timecourses to BIDS format.\n")
    
    # Validate input file exists
    csv_path_obj = Path(csv_path)
    if not csv_path_obj.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    if verbose:
        print(f"Input CSV: {csv_path_obj.name}")
        print(f"BIDS root: {bids_root}")
        print(f"Network label: {network_label}\n")
    
    # Call the main export function
    output_files = export_timecourses_to_bids(
        csv_path=csv_path,
        bids_root=bids_root,
        network_label=network_label,
        preamble_cols=preamble_cols,
        filename_prefix=filename_prefix,
        file_format=file_format,
        create_derivatives_subdir=create_derivatives_subdir,
        dry_run=dry_run,
        repetition_time=repetition_time,
        roi_metadata=roi_metadata,
        task_label=task_label,
        condition_to_task_mapping=condition_to_task_mapping,
        processing_description=processing_description,
        output_subdir=output_subdir,
        censor_convention=censor_convention,
    )
    
    # Print summary
    if verbose:
        print("=" * 80)
        total_files = sum(len(files) for files in output_files.values())
        
        if dry_run:
            print(f"✓ DRY-RUN COMPLETE: Would create {total_files} files for {len(output_files)} subjects")
        else:
            print(f"✓ EXPORT COMPLETE: Created {total_files} files for {len(output_files)} subjects")
        
        output_location = Path(bids_root) / "derivatives" / network_label
        print(f"\nOutput location: {output_location}/sub-XXXX/{output_subdir}/")
        print("=" * 80)
    
    return output_files


def compute_group_correlation_matrices(
    bids_root: str,
    network_label: str,
    subjects: Optional[List[str]] = None,
    tasks: Optional[List[str]] = None,
    intervention_label: Optional[str] = None,
    control_label: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Dict[str, Path]]:
    """
    Compute correlation matrices for all subjects in a BIDS dataset.
    
    For each subject and task combination, this function:
    1. Loads the timeseries data
    2. Removes censored timepoints
    3. Computes Pearson correlation matrix
    4. Applies Fisher z-transform
    5. Computes difference matrix (intervention - control) if both tasks present
    6. Saves outputs to {bids_root}/derivatives/fcmethods/
    
    Parameters
    ----------
    bids_root : str
        Root of the BIDS dataset
    network_label : str
        Network label (e.g., "cc200", "default")
    subjects : list, optional
        List of subject IDs to process (e.g., ["2002", "2003"]).
        If None, processes all subjects
    tasks : list, optional
        List of task labels to process (e.g., ["rest-intervention", "rest-control"]).
        If None, processes all available tasks
    intervention_label : str, optional
        Task label for intervention condition (e.g., "rest-drug").
        If None, no intervention matrix will be computed.
    control_label : str, optional
        Task label for control condition (e.g., "rest-placebo").
        If None, no control matrix will be computed.
    verbose : bool, optional
        If True, print detailed status messages. Default: True
    
    Returns
    -------
    output_files : dict
        Dictionary structure: {subject_id: {matrix_type: output_path}}
        matrix_type can be: "intervention", "control", "diff"
    
    Examples
    --------
    >>> output_files = compute_group_correlation_matrices(
    ...     bids_root="/path/to/BIDS",
    ...     network_label="cc200",
    ...     subjects=["2002", "2003"],
    ...     intervention_label="rest-drug",
    ...     control_label="rest-placebo"
    ... )
    """
    
    # Derive output root from BIDS structure
    bids_root_path = Path(bids_root)
    output_root = bids_root_path / "derivatives" / "fcmethods"
    
    if verbose:
        print("=" * 80)
        print("Computing Group Correlation Matrices")
        print("=" * 80)
        print(f"\nBIDS root: {bids_root}")
        print(f"Network label: {network_label}")
        print(f"Output location: {output_root}\n")
    
    # Find timeseries files in BIDS directory
    try:
        bids_files = get_bids_files(bids_root, network_label, subjects=subjects, tasks=tasks, verbose=verbose)
    except FileNotFoundError as e:
        print(f"✗ ERROR: {e}")
        raise
    
    if verbose:
        total_subs = len(bids_files)
        print(f"Found {total_subs} subject(s) with timeseries data")
    
    output_root.mkdir(parents=True, exist_ok=True)
    
    output_files = {}
    completed = 0
    
    # Process each subject
    for sub_id in sorted(bids_files.keys()):
        subject_tasks = bids_files[sub_id]
        
        if not subject_tasks:
            if verbose:
                search_dir = Path(bids_root) / "derivatives" / network_label / f"sub-{sub_id}"
                print(f"  ⊘ sub-{sub_id}: No timeseries files found in {search_dir}/")
            continue
        
        # Create output directory for this subject
        sub_output_dir = output_root / f"sub-{sub_id}"
        sub_output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Compute correlation matrices
            matrices = compute_subject_correlation_matrices(
                subject_tasks,
                output_dir=sub_output_dir,
                z_transform=True,
                compute_diff=True,
                intervention_label=intervention_label,
                control_label=control_label,
            )
            
            output_files[sub_id] = {
                matrix_type: sub_output_dir / f"corrmat_{matrix_type}.npy"
                for matrix_type in matrices.keys()
            }
            
            if verbose:
                matrix_types = ", ".join(matrices.keys())
                print(f"  ✓ sub-{sub_id}: {matrix_types}")
            
            completed += 1
            
        except Exception as e:
            if verbose:
                print(f"  ✗ sub-{sub_id}: {e}")
            continue
    
    # Summary
    if verbose:
        print("\n" + "=" * 80)
        print(f"✓ COMPLETE: Processed {completed}/{total_subs} subjects")
        print(f"Output location: {output_root}/sub-XXXX/corrmat_*.npy")
        print("=" * 80)
    
    return output_files


def visualize_correlation_matrices(
    output_root: str,
    subjects: Optional[List[str]] = None,
    roi_labels: Optional[List[str]] = None,
    roi_clusters: Optional[Dict[str, List[str]]] = None,
    matrix_display_names: Optional[Dict[str, str]] = None,
    participants_group_column: Optional[str] = None,
    participants_include_column: str = "include",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    group_symmetric_color_scale: bool = True,
    cmap: str = "RdBu_r",
    figsize: tuple = (15, 5),
    dpi: int = 150,
    verbose: bool = True,
) -> Dict[str, Path]:
    """
    Create and save heatmap visualizations for correlation matrices.
    
    Generates individual subject heatmaps and a group-averaged heatmap.
    Diagonal is removed (set to NaN) before plotting to avoid affecting color scale.
    
    Parameters
    ----------
    output_root : str
        Root directory containing sub-*/corrmat_*.npy files
    subjects : list, optional
        List of subject IDs to visualize (e.g., ["2002", "2003"]).
        If None, visualizes all subjects with correlation matrices
    roi_labels : list, optional
        Labels for ROIs (rows/columns of heatmaps). If None, attempts to infer
        labels from sub-*/corrmat_*.json (or cormat_*.json) sidecars via the
        "ROIs" field; if not found, falls back to numeric indices.
    roi_clusters : dict, optional
        ROI cluster definitions as {cluster_name: [roi1, roi2, ...]}.
        Draws dark boundary lines between adjacent ROIs that belong to
        different clusters. Hemisphere-prefixed ROI labels (e.g., L_roi, R_roi)
        are matched when clusters are defined without prefix.
    matrix_display_names : dict, optional
        Display-name mapping for matrix panels, e.g.
        {"intervention": "Drug", "control": "Placebo", "diff": "Drug - Placebo"}.
    participants_group_column : str, optional
        Column name in participants.tsv used to create additional group-average
        figures split by group value.
    participants_include_column : str, optional
        Column name in participants.tsv used for default subject filtering when
        subjects is None. If the column exists, only rows with value 1/true/yes
        are included. Default: "include".
    vmin, vmax : float, optional
        Color scale limits. If None, computed from data (1st and 99th percentiles)
    group_symmetric_color_scale : bool, optional
        If True, enforce a symmetric zero-centered color scale for the group
        figure so zero is at the white midpoint of a diverging colormap.
        Default: True.
    cmap : str, optional
        Colormap name (default: "RdBu_r")
    figsize : tuple, optional
        Figure size (width, height). Default: (15, 5)
    dpi : int, optional
        DPI for saved figures. Default: 150
    verbose : bool, optional
        If True, print progress messages. Default: True
    
    Returns
    -------
    output_files : dict
        Dictionary mapping "group" and subject IDs to paths of saved figures
    
    Examples
    --------
    >>> output_files = visualize_correlation_matrices(
    ...     output_root="/path/to/BIDS/derivatives/fcmethods",
    ...     roi_labels=["ROI1", "ROI2", "ROI3"],
    ...     cmap="RdBu_r"
    ... )
    >>> print(f"Saved {len(output_files)} visualizations")
    """
    
    if verbose:
        print("=" * 80)
        print("Visualizing Correlation Matrices")
        print("=" * 80)
        print(f"\nOutput root: {output_root}\n")
    
    output_root = Path(output_root)
    output_files = {}

    subject_dirs = sorted(output_root.glob("sub-*"))
    available_subjects = [d.name.replace("sub-", "") for d in subject_dirs]

    participants_group_assignments = {}
    used_participants_defaults = False
    used_include_filter = False

    # Find all subjects if not specified
    if subjects is None:
        (
            subjects,
            participants_group_assignments,
            used_include_filter,
            used_participants_defaults,
        ) = _get_default_subjects_and_groups(
            output_root=output_root,
            available_subjects=available_subjects,
            participants_group_column=participants_group_column,
            participants_include_column=participants_include_column,
        )
    else:
        subjects = [_normalize_subject_id(subject_id) for subject_id in subjects]
        participants_rows = _load_participants_rows(output_root)
        if participants_rows and participants_group_column is not None:
            for row in participants_rows:
                participant_id = row.get("participant_id")
                if not participant_id:
                    continue
                subject_id = _normalize_subject_id(participant_id)
                if subject_id not in subjects:
                    continue
                if participants_group_column in row:
                    group_value = str(row.get(participants_group_column, "")).strip()
                    if group_value:
                        participants_group_assignments[subject_id] = group_value

    # Auto-infer ROI labels from JSON sidecars if not provided
    if roi_labels is None:
        inferred_roi_labels = _infer_roi_labels_from_corrmat_json(output_root, subjects)
        if inferred_roi_labels is not None:
            roi_labels = inferred_roi_labels
            if verbose:
                print(f"Inferred ROI labels from JSON sidecar ({len(roi_labels)} labels)")
        elif verbose:
            print("Could not infer ROI labels from JSON sidecars; using numeric indices")
    
    if verbose:
        if used_participants_defaults:
            print("Resolved default subject list from participants.tsv")
        if used_include_filter:
            print(f"Applied participants.tsv filter: {participants_include_column} == 1")
        if participants_group_column is not None:
            print(f"Group-average split column: {participants_group_column}")
        print(f"Visualizing {len(subjects)} subject(s)")
    
    # Create individual subject visualizations
    completed = 0
    for sub_id in sorted(subjects):
        sub_dir = output_root / f"sub-{sub_id}"
        
        if not sub_dir.exists():
            if verbose:
                print(f"  ⊘ sub-{sub_id}: Directory not found")
            continue
        
        try:
            fig_path = visualize_subject_corrmat(
                subject_id=sub_id,
                output_dir=sub_dir,
                output_root=output_root,
                roi_labels=roi_labels,
                roi_clusters=roi_clusters,
                matrix_display_names=matrix_display_names,
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                figsize=figsize,
                dpi=dpi,
            )
            
            if fig_path is not None:
                output_files[f"sub-{sub_id}"] = fig_path
                if verbose:
                    print(f"  ✓ sub-{sub_id}: {fig_path.name}")
                completed += 1
            else:
                if verbose:
                    print(f"  ⊘ sub-{sub_id}: No correlation matrices found")
        
        except Exception as e:
            if verbose:
                print(f"  ✗ sub-{sub_id}: {e}")
            continue
    
    # Create group visualization
    try:
        group_fig_path = visualize_group_corrmat(
            output_root=output_root,
            subjects=subjects,
            roi_labels=roi_labels,
            roi_clusters=roi_clusters,
            matrix_display_names=matrix_display_names,
            symmetric_color_scale=group_symmetric_color_scale,
            cmap=cmap,
            figsize=figsize,
            dpi=dpi,
        )
        
        if group_fig_path is not None:
            output_files["group"] = group_fig_path
            if verbose:
                print(f"\n  ✓ Group average: {group_fig_path.name}")
    
    except Exception as e:
        if verbose:
            print(f"\n  ✗ Group average: {e}")

    # Create additional group-average visualizations split by participants.tsv group value
    grouped_average_count = 0
    if participants_group_column is not None:
        group_to_subjects = {}
        for subject_id in subjects:
            group_value = participants_group_assignments.get(subject_id)
            if not group_value:
                continue
            group_to_subjects.setdefault(group_value, []).append(subject_id)

        for group_value, group_subjects in sorted(group_to_subjects.items()):
            try:
                safe_group_value = _sanitize_group_value(group_value)
                group_key = f"group_{participants_group_column}_{safe_group_value}"
                group_fig_path = visualize_group_corrmat(
                    output_root=output_root,
                    subjects=group_subjects,
                    roi_labels=roi_labels,
                    roi_clusters=roi_clusters,
                    matrix_display_names=matrix_display_names,
                    symmetric_color_scale=group_symmetric_color_scale,
                    cmap=cmap,
                    figsize=figsize,
                    dpi=dpi,
                    output_filename=f"group_corrmat_heatmaps_{participants_group_column}-{safe_group_value}.png",
                    title_prefix=f"Group Average ({participants_group_column}={group_value}) ",
                )

                if group_fig_path is not None:
                    output_files[group_key] = group_fig_path
                    grouped_average_count += 1
                    if verbose:
                        print(
                            f"  ✓ Group average {participants_group_column}={group_value}: "
                            f"{group_fig_path.name}"
                        )
            except Exception as e:
                if verbose:
                    print(f"  ✗ Group average {participants_group_column}={group_value}: {e}")
    
    # Summary
    if verbose:
        print("\n" + "=" * 80)
        print(f"✓ COMPLETE: Created {completed} subject visualization(s)")
        if "group" in output_files:
            print(f"           + 1 group visualization")
        if grouped_average_count:
            print(f"           + {grouped_average_count} grouped average visualization(s)")
        print(f"\nOutput location: {output_root}/figures/")
        print("=" * 80)
    
    return output_files


def compute_graph_metrics_from_corrmats(
    output_root: str,
    subjects: Optional[List[str]] = None,
    matrix_types: Optional[List[str]] = None,
    threshold_mode: str = "cost",
    cost_thresholds: Optional[List[float]] = None,
    absolute_thresholds: Optional[List[float]] = None,
    quick: bool = False,
    positive_only: bool = True,
    participants_group_column: Optional[str] = None,
    participants_include_column: str = "include",
    save_adjacencies: bool = False,
    verbose: bool = True,
) -> Dict[str, Path]:
    """
    Compute CONN-style graph metrics from subject-level correlation matrices.

    Graphs are built per subject, per matrix type, and per threshold value using
    binary undirected adjacency matrices. Outputs are saved as TSV tables and
    JSON sidecars suitable for downstream figures/statistical analyses.

    Parameters
    ----------
    output_root : str
        Root directory containing sub-*/corrmat_*.npy files
    subjects : list, optional
        Subject IDs to include. If None, defaults to participants.tsv-based
        selection (include==1 when include column exists).
    matrix_types : list, optional
        Matrix types to process (e.g., ["intervention", "control"]).
        Default: ["intervention", "control"]
    threshold_mode : str, optional
        Thresholding mode: "cost" or "absolute". Default: "cost"
    cost_thresholds : list, optional
        Cost thresholds (0..1) for threshold_mode="cost".
        Default: [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    absolute_thresholds : list, optional
        Absolute thresholds for threshold_mode="absolute".
        Default: [0.5]
    quick : bool, optional
        If True, force a single 10% cost threshold for quick debugging.
        Default: False
    positive_only : bool, optional
        If True, only positive edges are eligible during thresholding.
        Default: True
    participants_group_column : str, optional
        Optional participants.tsv column to include group labels in outputs.
    participants_include_column : str, optional
        participants.tsv include filter column. Default: "include"
    save_adjacencies : bool, optional
        If True, saves per-subject adjacency matrices under graph/sub-*/.
        Default: False
    verbose : bool, optional
        Print progress messages. Default: True

    Returns
    -------
    output_files : dict
        Paths to generated TSV/JSON files.
    """
    if verbose:
        print("=" * 80)
        print("Computing Graph Metrics from Correlation Matrices")
        print("=" * 80)
        print(f"\nOutput root: {output_root}\n")

    output_root = Path(output_root)
    if not output_root.exists():
        raise FileNotFoundError(f"Output root does not exist: {output_root}")

    if matrix_types is None:
        matrix_types = ["intervention", "control"]

    subject_dirs = sorted(output_root.glob("sub-*"))
    available_subjects = [d.name.replace("sub-", "") for d in subject_dirs]

    participants_group_assignments = {}
    used_participants_defaults = False
    used_include_filter = False

    if subjects is None:
        (
            subjects,
            participants_group_assignments,
            used_include_filter,
            used_participants_defaults,
        ) = _get_default_subjects_and_groups(
            output_root=output_root,
            available_subjects=available_subjects,
            participants_group_column=participants_group_column,
            participants_include_column=participants_include_column,
        )
    else:
        subjects = [_normalize_subject_id(subject_id) for subject_id in subjects]
        participants_rows = _load_participants_rows(output_root)
        if participants_rows and participants_group_column is not None:
            for row in participants_rows:
                participant_id = row.get("participant_id")
                if not participant_id:
                    continue
                subject_id = _normalize_subject_id(participant_id)
                if subject_id not in subjects:
                    continue
                if participants_group_column in row:
                    group_value = str(row.get(participants_group_column, "")).strip()
                    if group_value:
                        participants_group_assignments[subject_id] = group_value

    if quick:
        threshold_mode = "cost"
        thresholds = [0.10]
    else:
        if threshold_mode == "cost":
            thresholds = cost_thresholds if cost_thresholds is not None else [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
        elif threshold_mode == "absolute":
            thresholds = absolute_thresholds if absolute_thresholds is not None else [0.5]
        else:
            raise ValueError("threshold_mode must be 'cost' or 'absolute'")

    thresholds = [float(t) for t in thresholds]
    thresholds = sorted(thresholds)

    participants_rows = _load_participants_rows(output_root)
    participants_meta = {}
    if participants_rows:
        for row in participants_rows:
            pid = row.get("participant_id")
            if not pid:
                continue
            sid = _normalize_subject_id(pid)
            participants_meta[sid] = row

    if verbose:
        if used_participants_defaults:
            print("Resolved default subject list from participants.tsv")
        if used_include_filter:
            print(f"Applied participants.tsv filter: {participants_include_column} == 1")
        print(f"Subjects included: {len(subjects)}")
        print(f"Matrix types: {', '.join(matrix_types)}")
        print(f"Threshold mode: {threshold_mode}")
        print(f"Thresholds: {thresholds}")
        if quick:
            print("Quick mode enabled: using single cost threshold of 0.10")

    node_rows = []
    network_rows = []

    total_processed = 0
    for subject_id in sorted(subjects):
        sub_dir = output_root / f"sub-{subject_id}"
        if not sub_dir.exists():
            if verbose:
                print(f"  ⊘ sub-{subject_id}: directory not found")
            continue

        participant_id = f"sub-{subject_id}"
        group_value = participants_group_assignments.get(subject_id)

        for matrix_type in matrix_types:
            matrix_path = sub_dir / f"corrmat_{matrix_type}.npy"
            if not matrix_path.exists():
                continue

            matrix = np.load(matrix_path)

            # Try to get ROI labels for node-level table
            roi_labels = None
            matrix_json = sub_dir / f"corrmat_{matrix_type}.json"
            if matrix_json.exists():
                try:
                    with open(matrix_json, "r") as f:
                        metadata = json.load(f)
                    rois = metadata.get("ROIs")
                    if isinstance(rois, list):
                        roi_labels = [str(x) for x in rois]
                except Exception:
                    roi_labels = None

            if roi_labels is None or len(roi_labels) != matrix.shape[0]:
                roi_labels = [f"ROI_{idx + 1}" for idx in range(matrix.shape[0])]

            for thr in thresholds:
                adjacency = build_adjacency_from_corrmat(
                    matrix=matrix,
                    threshold_mode=threshold_mode,
                    threshold_value=thr,
                    positive_only=positive_only,
                )

                node_df = compute_node_graph_metrics(adjacency)
                net_metrics = compute_network_graph_metrics(adjacency, node_metrics=node_df)

                for idx, row in node_df.iterrows():
                    out_row = {
                        "participant_id": participant_id,
                        "subject_id": subject_id,
                        "matrix_type": matrix_type,
                        "threshold_mode": threshold_mode,
                        "threshold_value": thr,
                        "quick": int(quick),
                        "roi_index": int(idx),
                        "roi": roi_labels[idx],
                    }
                    if participants_group_column is not None:
                        out_row[participants_group_column] = group_value

                    participant_meta = participants_meta.get(subject_id, {})
                    for meta_key, meta_value in participant_meta.items():
                        if meta_key not in out_row:
                            out_row[meta_key] = meta_value

                    for metric_name in node_df.columns:
                        out_row[metric_name] = row[metric_name]
                    node_rows.append(out_row)

                net_row = {
                    "participant_id": participant_id,
                    "subject_id": subject_id,
                    "matrix_type": matrix_type,
                    "threshold_mode": threshold_mode,
                    "threshold_value": thr,
                    "quick": int(quick),
                }
                if participants_group_column is not None:
                    net_row[participants_group_column] = group_value

                participant_meta = participants_meta.get(subject_id, {})
                for meta_key, meta_value in participant_meta.items():
                    if meta_key not in net_row:
                        net_row[meta_key] = meta_value

                net_row.update(net_metrics)
                network_rows.append(net_row)

                if save_adjacencies:
                    adjacency_dir = output_root / "graph" / f"sub-{subject_id}"
                    adjacency_dir.mkdir(parents=True, exist_ok=True)
                    thr_label = str(thr).replace(".", "p")
                    np.save(adjacency_dir / f"adjacency_{matrix_type}_{threshold_mode}-{thr_label}.npy", adjacency)

            total_processed += 1
            if verbose:
                print(f"  ✓ sub-{subject_id} {matrix_type}: {len(thresholds)} threshold(s)")

    node_df = pd.DataFrame(node_rows)
    network_df = pd.DataFrame(network_rows)

    if node_df.empty or network_df.empty:
        raise RuntimeError("No graph metrics were computed. Check matrix types and subject inputs.")

    metric_cols = [
        "degree",
        "cost",
        "avg_path_distance",
        "clustering_coefficient",
        "global_efficiency",
        "local_efficiency",
        "betweenness_centrality",
    ]

    node_auc_df = None
    network_auc_df = None
    if len(thresholds) > 1:
        node_group_cols = ["participant_id", "subject_id", "matrix_type", "roi_index", "roi"]
        network_group_cols = ["participant_id", "subject_id", "matrix_type"]
        if participants_group_column is not None and participants_group_column in node_df.columns:
            node_group_cols.append(participants_group_column)
        if participants_group_column is not None and participants_group_column in network_df.columns:
            network_group_cols.append(participants_group_column)

        node_auc_df = compute_auc_by_group(
            df=node_df,
            threshold_col="threshold_value",
            metric_cols=metric_cols,
            group_cols=node_group_cols,
        )
        network_auc_df = compute_auc_by_group(
            df=network_df,
            threshold_col="threshold_value",
            metric_cols=metric_cols,
            group_cols=network_group_cols,
        )

    metadata = {
        "Description": "Graph metrics computed from ROI-to-ROI correlation matrices",
        "SourceOutputRoot": str(output_root),
        "ThresholdMode": threshold_mode,
        "ThresholdValues": thresholds,
        "QuickMode": bool(quick),
        "PositiveOnlyEdges": bool(positive_only),
        "MatrixTypes": matrix_types,
        "ParticipantsIncludeColumn": participants_include_column,
        "ParticipantsGroupColumn": participants_group_column,
        "GeneratedAt": datetime.now().isoformat(),
        "Metrics": {
            "degree": "Number of supra-threshold edges connected to each node",
            "cost": "Node degree normalized by N-1",
            "avg_path_distance": "Average shortest-path distance to reachable nodes",
            "clustering_coefficient": "Local neighborhood edge density",
            "global_efficiency": "Average inverse shortest-path distance to all other nodes",
            "local_efficiency": "Global efficiency of node neighborhood subgraph",
            "betweenness_centrality": "Proportion of shortest paths passing through each node",
        },
    }

    graph_output_dir = output_root / "graph"
    output_files = save_graph_outputs(
        output_dir=graph_output_dir,
        node_df=node_df,
        network_df=network_df,
        node_auc_df=node_auc_df,
        network_auc_df=network_auc_df,
        metadata=metadata,
    )

    if verbose:
        print("\n" + "=" * 80)
        print(f"✓ COMPLETE: Computed graph metrics for {total_processed} subject/matrix set(s)")
        print(f"Output location: {graph_output_dir}")
        for key, value in output_files.items():
            print(f"  {key}: {value.name}")
        print("=" * 80)

    return output_files
