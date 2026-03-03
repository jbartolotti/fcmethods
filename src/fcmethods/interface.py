"""
User-facing interface for timecourse export and network analysis with logging and reporting.

This module provides high-level functions with comprehensive console output and
error handling for both timecourse export and correlation matrix computation.
"""

from pathlib import Path
from typing import Optional, List, Dict

from .timecourse_io import export_timecourses_to_bids
from .network_analysis import (
    get_bids_files,
    compute_subject_correlation_matrices,
)


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
        bids_files = get_bids_files(bids_root, network_label, subjects=subjects, tasks=tasks)
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
                print(f"  ⊘ sub-{sub_id}: No timeseries files found")
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
        print("=" * 80)
    
    return output_files
