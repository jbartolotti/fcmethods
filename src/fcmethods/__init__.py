"""
fcmethods: Functional Connectivity Analysis Methods for BIDS
"""

from .timecourse_io import parse_timecourse_csv, export_timecourses_to_bids

__version__ = "0.1.0"
__all__ = ["parse_timecourse_csv", "export_timecourses_to_bids"]
