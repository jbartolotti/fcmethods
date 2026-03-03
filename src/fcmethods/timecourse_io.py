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
    filename_prefix: str = "timecourse",
    file_format: str = "tsv",
    create_derivatives_subdir: bool = True,
    dry_run: bool = False,
) -> Dict[str, List[str]]:
    """
    Export timecourse data from CSV to BIDS-compliant derivative structure.
    
    Creates the following directory structure:
    {bids_root}/derivatives/{network_label}/sub-{subnum}/timeseries/
    
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
        Prefix for output files. Default: "timecourse"
    file_format : str, optional
        Output format: "tsv", "csv", or "npy". Default: "tsv"
    create_derivatives_subdir : bool, optional
        If True, organize by derivatives/{network_label}/sub-{subnum}
        If False, organize by derivatives/sub-{subnum}
    dry_run : bool, optional
        If True, preview the output without writing files. Default: False
    
    Returns
    -------
    output_files : dict
        Dictionary mapping subject IDs to lists of output file paths
    
    Examples
    --------
    >>> output_files = export_timecourses_to_bids(
    ...     csv_path="timecourses.csv",
    ...     bids_root="/path/to/BIDS",
    ...     network_label="fcsanity"
    ... )
    
    >>> # Dry-run to preview what would be written
    >>> output_files = export_timecourses_to_bids(
    ...     csv_path="timecourses.csv",
    ...     bids_root="/path/to/BIDS",
    ...     network_label="fcsanity",
    ...     dry_run=True
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
    
    # Group by subject
    for subnum, group_data in preamble.groupby("subnum"):
        # Get indices for this subject
        indices = group_data.index
        
        # Get relevant timecourses for this subject
        subject_timecourses = timecourse_data.iloc[indices].copy()
        subject_preamble = preamble.iloc[indices].copy()
        
        # Create output directory
        if create_derivatives_subdir:
            output_dir = bids_root / "derivatives" / network_label / f"sub-{subnum}" / "timeseries"
        else:
            output_dir = bids_root / "derivatives" / f"sub-{subnum}" / "timeseries"
        
        if not dry_run:
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare output data (combine preamble and timecourses)
        output_data = pd.concat([subject_preamble.reset_index(drop=True), 
                                  subject_timecourses.reset_index(drop=True)], 
                                 axis=1)
        
        # Generate filename
        if file_format in ["tsv", "csv"]:
            delimiter = "\t" if file_format == "tsv" else ","
            output_file = output_dir / f"sub-{subnum}_{filename_prefix}.{file_format}"
            
            if not dry_run:
                output_data.to_csv(output_file, sep=delimiter, index=False, na_rep="n/a")
            else:
                print(f"  {output_file}")
            
            output_files[str(subnum)] = [str(output_file)]
            
        elif file_format == "npy":
            # Save as numpy array (only timecourse data)
            output_file = output_dir / f"sub-{subnum}_{filename_prefix}.npy"
            metadata_file = output_dir / f"sub-{subnum}_{filename_prefix}_info.csv"
            
            if not dry_run:
                np.save(output_file, subject_timecourses.values)
                
                # Also save a metadata file with column names and preamble
                info_data = pd.DataFrame({
                    "roi": timecourse_cols,
                })
                info_data.to_csv(metadata_file, index=False)
            else:
                print(f"  {output_file}")
                print(f"  {metadata_file}")
            
            output_files[str(subnum)] = [str(output_file), str(metadata_file)]
        
        else:
            raise ValueError(f"Unsupported file_format: {file_format}. Use 'tsv', 'csv', or 'npy'.")
    
    if dry_run:
        print(f"\n[DRY RUN MODE] Would export timecourses for {len(output_files)} subjects.")
    
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
