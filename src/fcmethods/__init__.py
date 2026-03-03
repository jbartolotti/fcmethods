"""
fcmethods: Functional Connectivity Analysis Methods for BIDS
"""

from .timecourse_io import (
    parse_timecourse_csv,
    export_timecourses_to_bids,
)
from .interface import (
    export_timecourses_to_bids_with_reporting,
    compute_group_correlation_matrices,
)
from .network_analysis import (
    fisher_z_transform,
    inverse_fisher_z_transform,
    load_timeseries_from_bids,
    compute_correlation_matrix,
    get_bids_files,
    compute_subject_correlation_matrices,
)

__version__ = "0.1.0"
__all__ = [
    "parse_timecourse_csv",
    "export_timecourses_to_bids",
    "export_timecourses_to_bids_with_reporting",
    "fisher_z_transform",
    "inverse_fisher_z_transform",
    "load_timeseries_from_bids",
    "compute_correlation_matrix",
    "get_bids_files",
    "compute_subject_correlation_matrices",
    "compute_group_correlation_matrices",
]
