"""
Timecourse I/O utilities for reading and writing functional connectivity data.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Dict


def parse_timecourse_csv(
    csv_path: str,
    preamble_cols: Optional[List[str]] = None,
    na_values: List = None,
) -> Tuple[pd.DataFrame, List[str], pd.DataFrame]:
    """
    Parse a timecourse CSV file with preamble columns and timecourse data.
    
    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing timecourse data
    preamble_cols : list, optional
        Column names to treat as preamble (metadata) columns.
        Default: ["slicenum", "condition", "subnum", "run", "time", "censor", "subgroup"]
    na_values : list, optional
        Values to treat as NA/NaN. Default: [None, 'NA', 'nan', '']
    
    Returns
    -------
    preamble : pd.DataFrame
        DataFrame containing preamble columns
    timecourse_cols : list
        List of column names that are timecourses (ROI/region labels)
    timecourse_data : pd.DataFrame
        DataFrame containing timecourse data for each ROI
    """
    
    if preamble_cols is None:
        preamble_cols = ["slicenum", "condition", "subnum", "run", "time", "censor", "subgroup"]
    
    if na_values is None:
        na_values = [None, "NA", "nan", "", "N/A"]
    
    # Read CSV with appropriate NA handling
    df = pd.read_csv(csv_path, na_values=na_values)
    
    # Identify which columns are preamble (metadata) vs timecourse data
    available_preamble_cols = [col for col in preamble_cols if col in df.columns]
    preamble = df[available_preamble_cols].copy()
    
    # Timecourse columns are everything else
    timecourse_cols = [col for col in df.columns if col not in available_preamble_cols]
    timecourse_data = df[timecourse_cols].copy()
    
    return preamble, timecourse_cols, timecourse_data


def export_timecourses_to_bids(
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
) -> Dict[str, List[str]]:
    """
    Export timecourse data from CSV to BIDS-compliant derivative structure.
    
    Creates BIDS-compliant TSV files with minimal columns (timepoint, censor, ROI timecourses)
    and JSON sidecars containing metadata.
    
    Directory structure:
    {bids_root}/derivatives/{network_label}/sub-{subnum}/[ses-{session}/]func/
    
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
        If False, organize by derivatives/sub-{subnum}
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
    
    Returns
    -------
    output_files : dict
        Dictionary mapping subject IDs to lists of output file paths (TSV and JSON)
    
    Examples
    --------
    >>> output_files = export_timecourses_to_bids(
    ...     csv_path="timecourses.csv",
    ...     bids_root="/path/to/BIDS",
    ...     network_label="fcsanity",
    ...     repetition_time=2.0,
    ...     task_label="rest"
    ... )
    
    >>> # With ROI metadata
    >>> roi_meta = {
    ...     "L_lPFC": {"hemisphere": "left", "radius_mm": 6},
    ...     "R_lPFC": {"hemisphere": "right", "radius_mm": 6}
    ... }
    >>> output_files = export_timecourses_to_bids(
    ...     csv_path="timecourses.csv",
    ...     bids_root="/path/to/BIDS",
    ...     network_label="fcsanity",
    ...     repetition_time=2.0,
    ...     roi_metadata=roi_meta,
    ...     dry_run=True
    ... )
    
    >>> # With condition-to-task mapping
    >>> output_files = export_timecourses_to_bids(
    ...     csv_path="timecourses.csv",
    ...     bids_root="/path/to/BIDS",
    ...     network_label="fcsanity",
    ...     repetition_time=2.0,
    ...     condition_to_task_mapping={"A": "rest-NTX", "B": "rest-PCB"}
    ... )
    """
    
    # Parse the CSV
    preamble, timecourse_cols, timecourse_data = parse_timecourse_csv(
        csv_path, preamble_cols=preamble_cols
    )
    
    # Validate that we have subnum column
    if "subnum" not in preamble.columns:
        raise ValueError("'subnum' column not found in preamble. Check preamble_cols.")
    
    output_files = {}
    bids_root = Path(bids_root)
    
    if dry_run:
        print("[DRY RUN MODE] The following files would be written:\n")
    
    # Track if we've printed a sample preview
    sample_printed = False
    sample_json_printed = False
    
    # Group by subject
    for subnum, group_data in preamble.groupby("subnum"):
        # Get indices for this subject
        indices = group_data.index
        
        # Get relevant timecourses for this subject
        subject_timecourses = timecourse_data.iloc[indices].copy()
        subject_preamble = preamble.iloc[indices].copy()
        
        # Combine for easier grouping
        full_data = pd.concat([subject_preamble.reset_index(drop=True), 
                               subject_timecourses.reset_index(drop=True)], 
                              axis=1)
        
        # Group by session and condition for file output
        groupby_cols = []
        if "run" in full_data.columns:
            groupby_cols.append("run")
        if "condition" in full_data.columns:
            groupby_cols.append("condition")
        
        if not groupby_cols:
            # No session or condition, treat all data as one file
            groups = [(None, full_data)]
        else:
            groups = list(full_data.groupby(groupby_cols, dropna=False))
        
        for group_key, session_cond_data in groups:
            # Handle single vs multiple groupby columns
            if groupby_cols:
                if len(groupby_cols) == 1:
                    session, condition = (group_key, None) if groupby_cols[0] == "run" else (None, group_key)
                else:
                    session, condition = group_key
            else:
                session, condition = None, None
            
            # Build output directory
            if create_derivatives_subdir:
                output_dir = bids_root / "derivatives" / network_label / f"sub-{subnum}"
                if session is not None and pd.notna(session):
                    output_dir = output_dir / f"ses-{session}"
                output_dir = output_dir / output_subdir
            else:
                output_dir = bids_root / "derivatives" / f"sub-{subnum}" / output_subdir
            
            if not dry_run:
                output_dir.mkdir(parents=True, exist_ok=True)
            
            # Build BIDS-compliant filename
            filename_parts = [f"sub-{subnum}"]
            
            # Add session if available and not null
            if session is not None and pd.notna(session):
                filename_parts.append(f"ses-{session}")
            
            # Determine task label: use condition_to_task_mapping if provided, else use task_label
            effective_task = None
            if condition_to_task_mapping is not None and condition is not None and pd.notna(condition):
                # Map condition to task
                condition_str = str(condition)
                if condition_str in condition_to_task_mapping:
                    effective_task = condition_to_task_mapping[condition_str]
            
            if effective_task is None and task_label is not None:
                effective_task = task_label
            
            # Add task if available
            if effective_task is not None:
                filename_parts.append(f"task-{effective_task}")
            
            # Add condition as desc if available and not null (and not already mapped to task)
            if condition is not None and pd.notna(condition) and condition_to_task_mapping is None:
                # Only use condition as desc if we're not using condition_to_task_mapping
                # Clean condition name (remove spaces, special chars)
                clean_cond = str(condition).replace(" ", "").replace("_", "")
                filename_parts.append(f"desc-{clean_cond}")
            
            filename_parts.append(filename_prefix)
            base_filename = "_".join(filename_parts)
            
            # Prepare BIDS-compliant TSV (only essential columns + timecourses)
            tsv_columns = []
            
            # Include timepoint/slicenum if available
            if "slicenum" in session_cond_data.columns:
                tsv_columns.append("slicenum")
            
            # Include censor column if available
            if "censor" in session_cond_data.columns:
                tsv_columns.append("censor")
            
            # Add all timecourse columns
            tsv_columns.extend(timecourse_cols)
            
            # Extract only BIDS-compliant columns for TSV
            tsv_data = session_cond_data[tsv_columns].copy()
            
            # Prepare JSON sidecar metadata
            json_metadata = {
                "Description": f"Timeseries data extracted from {network_label} ROIs",
                "Sources": [str(Path(csv_path).name)],
            }
            
            # Add repetition time if provided
            if repetition_time is not None:
                json_metadata["RepetitionTime"] = repetition_time
                json_metadata["SamplingFrequency"] = 1.0 / repetition_time
            
            # Add processing description if provided
            if processing_description is not None:
                json_metadata["ProcessingDescription"] = processing_description
            
            # Add column descriptions
            json_metadata["Columns"] = {}
            if "slicenum" in tsv_columns:
                json_metadata["Columns"]["slicenum"] = "Timepoint index (0-based or 1-based)"
            if "censor" in tsv_columns:
                if censor_convention == "inverted":
                    json_metadata["Columns"]["censor"] = "Retention flag (1=retained, 0=censored/excluded)"
                else:
                    json_metadata["Columns"]["censor"] = "Censoring flag (1=censored/excluded, 0=retained)"
            
            for roi in timecourse_cols:
                json_metadata["Columns"][roi] = f"Mean timeseries from {roi} ROI"
            
            # Add ROI metadata if provided
            if roi_metadata is not None:
                json_metadata["ROIs"] = roi_metadata
            else:
                # Create basic ROI metadata
                json_metadata["ROIs"] = {roi: {"label": roi} for roi in timecourse_cols}
            
            # Add additional metadata from preamble (as dataset-level info)
            json_metadata["Metadata"] = {}
            if session is not None and pd.notna(session):
                json_metadata["Metadata"]["session"] = str(session)
            if condition is not None and pd.notna(condition):
                json_metadata["Metadata"]["condition"] = str(condition)
            
            # Generate filenames
            if file_format in ["tsv", "csv"]:
                delimiter = "\t" if file_format == "tsv" else ","
                output_file = output_dir / f"{base_filename}.{file_format}"
                json_file = output_dir / f"{base_filename}.json"
                
                if not dry_run:
                    # Write TSV
                    tsv_data.to_csv(output_file, sep=delimiter, index=False, na_rep="n/a")
                    
                    # Write JSON sidecar
                    import json
                    with open(json_file, 'w') as f:
                        json.dump(json_metadata, f, indent=2)
                else:
                    print(f"  {output_file}")
                    print(f"  {json_file}")
                    
                    # Print sample data for the first file in dry-run
                    if not sample_printed:
                        print(f"\n    Sample TSV content (first 2 rows):")
                        sample_lines = tsv_data.head(2)
                        print(f"      Columns: {list(tsv_data.columns)}")
                        for idx, row in sample_lines.iterrows():
                            row_preview = {k: v for k, v in row.items()}
                            print(f"      Row {idx}: {row_preview}")
                        sample_printed = True
                    
                    # Print sample JSON for the first file
                    if not sample_json_printed:
                        print(f"\n    Sample JSON sidecar content:")
                        import json
                        print("      " + json.dumps(json_metadata, indent=6).replace("\n", "\n      "))
                        sample_json_printed = True
                        print()
                
                if str(subnum) not in output_files:
                    output_files[str(subnum)] = []
                output_files[str(subnum)].append(str(output_file))
                output_files[str(subnum)].append(str(json_file))
                
            elif file_format == "npy":
                # Save as numpy array (only timecourse data, no slicenum/censor)
                output_file = output_dir / f"{base_filename}.npy"
                json_file = output_dir / f"{base_filename}.json"
                
                # For npy, save only the timecourse columns
                timecourse_array = tsv_data[timecourse_cols].values
                
                if not dry_run:
                    np.save(output_file, timecourse_array)
                    
                    # Write JSON sidecar with all metadata
                    import json
                    with open(json_file, 'w') as f:
                        json.dump(json_metadata, f, indent=2)
                else:
                    print(f"  {output_file}")
                    print(f"  {json_file}")
                
                if str(subnum) not in output_files:
                    output_files[str(subnum)] = []
                output_files[str(subnum)].append(str(output_file))
                output_files[str(subnum)].append(str(json_file))
            
            else:
                raise ValueError(f"Unsupported file_format: {file_format}. Use 'tsv', 'csv', or 'npy'.")
    
    if dry_run:
        total_files = sum(len(files) for files in output_files.values())
        print(f"[DRY RUN MODE] Would export {total_files} files for {len(output_files)} subjects.")
    
    return output_files


def load_bids_timecourse(
    bids_root: str,
    subnum: str,
    network_label: str,
    filename_prefix: str = "timecourse",
    file_format: str = "tsv",
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load timecourse data that was exported with export_timecourses_to_bids.
    
    Parameters
    ----------
    bids_root : str
        Root path of the BIDS dataset
    subnum : str
        Subject number (without 'sub-' prefix)
    network_label : str
        Network/parcellation label
    filename_prefix : str, optional
        Prefix used when exporting. Default: "timecourse"
    file_format : str, optional
        Format of the file. Default: "tsv"
    
    Returns
    -------
    data : pd.DataFrame or np.ndarray
        The timecourse data
    timecourse_cols : list
        Names of the timecourse columns
    """
    
    bids_root = Path(bids_root)
    
    if file_format in ["tsv", "csv"]:
        delimiter = "\t" if file_format == "tsv" else ","
        data_file = bids_root / "derivatives" / network_label / f"sub-{subnum}" / "timeseries" / f"sub-{subnum}_{filename_prefix}.{file_format}"
        
        data = pd.read_csv(data_file, sep=delimiter, na_values=["n/a", "NA"])
        
        # Identify timecourse columns (numeric columns that are not metadata)
        preamble_cols = ["slicenum", "condition", "subnum", "run", "time", "censor", "subgroup"]
        timecourse_cols = [col for col in data.columns if col not in preamble_cols]
        
        return data, timecourse_cols
    
    elif file_format == "npy":
        data_file = bids_root / "derivatives" / network_label / f"sub-{subnum}" / "timeseries" / f"sub-{subnum}_{filename_prefix}.npy"
        info_file = bids_root / "derivatives" / network_label / f"sub-{subnum}" / "timeseries" / f"sub-{subnum}_{filename_prefix}_info.csv"
        
        data = np.load(data_file)
        info = pd.read_csv(info_file)
        timecourse_cols = info["roi"].tolist()
        
        return data, timecourse_cols
    
    else:
        raise ValueError(f"Unsupported file_format: {file_format}. Use 'tsv', 'csv', or 'npy'.")

