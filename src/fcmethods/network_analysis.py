"""
Network analysis utilities: correlation matrices and connectivity analysis.

This module provides functions to compute and analyze functional connectivity
from BIDS-formatted timeseries data, including correlation matrices, 
Fisher z-transforms, and group-level analyses.
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Optional, Dict, Tuple, List
from scipy.stats import fisher_exact


def fisher_z_transform(r: np.ndarray) -> np.ndarray:
    """
    Apply Fisher z-transform to correlation coefficients.
    
    Transforms r to z using: z = 0.5 * ln((1+r)/(1-r))
    Handles edge cases where |r| >= 1.
    
    Parameters
    ----------
    r : np.ndarray
        Correlation coefficients (should be in [-1, 1])
    
    Returns
    -------
    z : np.ndarray
        Fisher z-transformed values
    """
    # Clip values to avoid log(0) and log(negative)
    r_clipped = np.clip(r, -0.9999, 0.9999)
    z = 0.5 * np.log((1 + r_clipped) / (1 - r_clipped))
    return z


def inverse_fisher_z_transform(z: np.ndarray) -> np.ndarray:
    """
    Inverse Fisher z-transform back to correlation space.
    
    Transforms z back to r using: r = (e^(2z) - 1) / (e^(2z) + 1)
    
    Parameters
    ----------
    z : np.ndarray
        Fisher z-transformed values
    
    Returns
    -------
    r : np.ndarray
        Correlation coefficients
    """
    r = (np.exp(2*z) - 1) / (np.exp(2*z) + 1)
    return r


def load_timeseries_from_bids(
    timeseries_file: str,
    json_file: Optional[str] = None,
    remove_censored: bool = True,
) -> Tuple[np.ndarray, List[str], Dict]:
    """
    Load timeseries data from a BIDS-formatted TSV file with JSON metadata.
    
    Parameters
    ----------
    timeseries_file : str
        Path to the timeseries TSV file
    json_file : str, optional
        Path to the corresponding JSON sidecar. If None, will look for
        a file with the same name but .json extension
    remove_censored : bool, optional
        If True, remove censored timepoints (censor=0 for standard, 1 for inverted).
        Default: True
    
    Returns
    -------
    timeseries : np.ndarray
        Timeseries data (timepoints x ROIs)
    roi_labels : list
        Names of the ROIs (columns)
    metadata : dict
        Metadata from the JSON sidecar
    """
    
    # Read TSV
    ts_path = Path(timeseries_file)
    data = pd.read_csv(ts_path, sep='\t', na_values=['n/a', 'NA'])
    
    # Load JSON metadata if provided or found
    if json_file is None:
        json_file = ts_path.with_suffix('.json')
    
    metadata = {}
    censor_convention = "standard"  # Default
    if json_file and Path(json_file).exists():
        with open(json_file, 'r') as f:
            metadata = json.load(f)
        
        # Extract censor convention from metadata
        if "Columns" in metadata and "censor" in metadata["Columns"]:
            censor_desc = metadata["Columns"]["censor"]
            if "1=retained" in censor_desc or "retained" in censor_desc:
                censor_convention = "inverted"
    
    # Identify metadata columns (columns that are NOT timeseries data)
    metadata_cols = ["slicenum", "time", "condition", "censor", "subnum", "run", "subgroup"]
    data_cols = [col for col in data.columns if col not in metadata_cols]
    
    # Extract timeseries
    timeseries_data = data[data_cols].copy()
    
    # Remove censored timepoints if requested
    if remove_censored and "censor" in data.columns:
        if censor_convention == "inverted":
            # inverted: 1=retained, 0=censored -> keep where censor==1
            mask = data["censor"] == 1
        else:
            # standard: 1=censored, 0=retained -> keep where censor==0
            mask = data["censor"] == 0
        
        timeseries_data = timeseries_data[mask].reset_index(drop=True)
    
    # Convert to numpy and handle NaNs
    ts_array = timeseries_data.values.astype(float)
    
    # Remove any remaining NaN values (across timepoints/columns)
    # Keep only complete timepoints
    valid_mask = ~np.any(np.isnan(ts_array), axis=1)
    ts_array = ts_array[valid_mask]
    
    roi_labels = data_cols
    
    return ts_array, roi_labels, metadata


def compute_correlation_matrix(
    timeseries: np.ndarray,
    method: str = "pearson",
) -> np.ndarray:
    """
    Compute correlation matrix from timeseries data.
    
    Parameters
    ----------
    timeseries : np.ndarray
        Timeseries data (timepoints x ROIs)
    method : str, optional
        Correlation method: "pearson", "spearman", "kendall". Default: "pearson"
    
    Returns
    -------
    corrmat : np.ndarray
        Correlation matrix (ROIs x ROIs)
    """
    
    if method == "pearson":
        corrmat = np.corrcoef(timeseries.T)
    elif method == "spearman":
        from scipy.stats import spearmanr
        corrmat = spearmanr(timeseries)[0]  # Returns correlation matrix
    elif method == "kendall":
        from scipy.stats import kendalltau
        n_rois = timeseries.shape[1]
        corrmat = np.zeros((n_rois, n_rois))
        for i in range(n_rois):
            for j in range(n_rois):
                corrmat[i, j] = kendalltau(timeseries[:, i], timeseries[:, j])[0]
    else:
        raise ValueError(f"Unknown method: {method}. Use 'pearson', 'spearman', or 'kendall'.")
    
    return corrmat


def get_bids_files(
    bids_root: str,
    network_label: str,
    subjects: Optional[List[str]] = None,
    tasks: Optional[List[str]] = None,
    verbose: bool = False,
) -> Dict[str, Dict[str, Path]]:
    """
    Find timeseries files in a BIDS directory.
    
    Parameters
    ----------
    bids_root : str
        Root of the BIDS dataset
    network_label : str
        Network label (e.g., "cc200", "default")
    subjects : list, optional
        List of subject IDs to include (e.g., ["2002", "2003"]). 
        If None, includes all subjects
    tasks : list, optional
        List of task labels to include (e.g., ["rest-drug", "rest-placebo"]).
        If None, includes all tasks
    verbose : bool, optional
        If True, print debug info about file discovery. Default: False
    
    Returns
    -------
    files : dict
        Dictionary structure: {subject_id: {task_label: Path}}
    """
    
    bids_root = Path(bids_root)
    derivatives_dir = bids_root / "derivatives" / network_label
    
    if not derivatives_dir.exists():
        raise FileNotFoundError(f"Derivatives directory not found: {derivatives_dir}")
    
    files = {}
    
    # Find all subject directories
    for sub_dir in sorted(derivatives_dir.glob("sub-*")):
        sub_id = sub_dir.name.replace("sub-", "")
        
        # Filter by subjects if specified
        if subjects is not None and sub_id not in subjects:
            continue
        
        files[sub_id] = {}
        
        # Find all timeseries files in func subdirectory (recursive to handle sessions)
        found_any = False
        for ts_file in sorted(sub_dir.glob("**/*stat-mean_timeseries.tsv")):
            found_any = True
            # Extract task from filename
            filename = ts_file.stem
            # Assuming filename like: sub-XXX_task-rest-drug_desc-XXX_stat-mean_timeseries
            parts = filename.split("_")
            task = None
            for part in parts:
                if part.startswith("task-"):
                    task = part.replace("task-", "")
                    break
            
            if task is None:
                if verbose:
                    print(f"  DEBUG: sub-{sub_id}: Found TSV but could not extract task from {ts_file.name}")
                continue
            
            # Filter by tasks if specified
            if tasks is not None and task not in tasks:
                continue
            
            files[sub_id][task] = ts_file
        
        if verbose and not found_any:
            print(f"  DEBUG: sub-{sub_id}: No TSV files matching pattern **/stat-mean_timeseries.tsv in {sub_dir}")
    
    return files


def compute_subject_correlation_matrices(
    timeseries_files: Dict[str, Path],
    output_dir: Optional[Path] = None,
    z_transform: bool = True,
    compute_diff: bool = True,
    intervention_label: Optional[str] = None,
    control_label: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """
    Compute correlation matrices for a single subject across tasks.
    
    Parameters
    ----------
    timeseries_files : dict
        Dictionary mapping task labels to timeseries file paths
    output_dir : Path, optional
        If provided, save correlation matrices to this directory
    z_transform : bool, optional
        Apply Fisher z-transform. Default: True
    compute_diff : bool, optional
        Compute difference matrix (intervention - control). Default: True
    intervention_label : str, optional
        Task label for intervention condition (e.g., "rest-drug").
        If None, no intervention matrix will be computed.
    control_label : str, optional
        Task label for control condition (e.g., "rest-placebo").
        If None, no control matrix will be computed.
    
    Returns
    -------
    matrices : dict
        Dictionary with keys: "intervention", "control", "diff" (if compute_diff=True)
        Each value is the correlation matrix (ROIs x ROIs)
    """
    
    matrices = {}
    roi_labels = None
    
    # Compute correlation matrices for each task
    for task_label, ts_file in timeseries_files.items():
        timeseries, roi_labels, _ = load_timeseries_from_bids(ts_file)
        corrmat = compute_correlation_matrix(timeseries)
        
        if z_transform:
            corrmat = fisher_z_transform(corrmat)
        
        # Map to standardized keys
        if intervention_label is not None and task_label == intervention_label:
            matrices["intervention"] = corrmat
        elif control_label is not None and task_label == control_label:
            matrices["control"] = corrmat
        else:
            matrices[task_label] = corrmat
    
    # Compute difference matrix if both conditions are available
    if compute_diff and "intervention" in matrices and "control" in matrices:
        matrices["diff"] = matrices["intervention"] - matrices["control"]
    
    # Save if output directory provided
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for matrix_type, matrix in matrices.items():
            np.save(output_dir / f"corrmat_{matrix_type}.npy", matrix)
            
            # Save metadata JSON
            metadata = {
                "MatrixType": matrix_type,
                "Shape": list(matrix.shape),
                "ZTransformed": z_transform,
                "ROIs": roi_labels,
            }
            with open(output_dir / f"corrmat_{matrix_type}.json", 'w') as f:
                json.dump(metadata, f, indent=2)
    
    return matrices
