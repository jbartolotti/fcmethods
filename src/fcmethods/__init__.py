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
    visualize_correlation_matrices,
    compute_graph_metrics_from_corrmats,
)
from .network_analysis import (
    fisher_z_transform,
    inverse_fisher_z_transform,
    load_timeseries_from_bids,
    compute_correlation_matrix,
    get_bids_files,
    compute_subject_correlation_matrices,
)
from .visualization import (
    plot_correlation_matrices,
    visualize_subject_corrmat,
    visualize_group_corrmat,
    remove_diagonal,
)
from .graph_analysis import (
    build_adjacency_from_corrmat,
    compute_node_graph_metrics,
    compute_network_graph_metrics,
    compute_auc_by_group,
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
    "compute_graph_metrics_from_corrmats",
    "plot_correlation_matrices",
    "visualize_subject_corrmat",
    "visualize_group_corrmat",
    "visualize_correlation_matrices",
    "remove_diagonal",
    "build_adjacency_from_corrmat",
    "compute_node_graph_metrics",
    "compute_network_graph_metrics",
    "compute_auc_by_group",
]
