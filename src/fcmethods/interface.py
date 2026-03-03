"""
User-facing interface for timecourse export with logging and reporting.

This module provides high-level functions for exporting timecourses with
comprehensive console output and error handling.
"""

from pathlib import Path
from typing import Optional, List, Dict

from .timecourse_io import export_timecourses_to_bids


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
        Network/parcellation label (e.g., "fcsanity", "schaefer400")
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
        {"A": "rest-NTX", "B": "rest-PCB"}
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
    ...     network_label="fcsanity",
    ...     repetition_time=2.0,
    ...     condition_to_task_mapping={"A": "rest-NTX", "B": "rest-PCB"},
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
