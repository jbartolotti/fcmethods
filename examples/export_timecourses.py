"""
Example: Export timecourses to BIDS format

This example shows how to use fcmethods to parse timecourse data from a CSV
and export it to a BIDS-compliant derivative structure with JSON sidecars.
"""

from fcmethods import export_timecourses_to_bids

# Paths
csv_path = r"C:\path\to\timecourses.csv"
bids_root = r"C:\path\to\BIDS"

# Optional: Define ROI metadata for JSON sidecars
roi_metadata = {
    "L_lPFC": {
        "hemisphere": "left",
        "region": "lateral prefrontal cortex",
        "radius_mm": 6,
        # "center_xyz": [-44, 36, 20]  # Optional MNI coordinates
    },
    "R_lPFC": {
        "hemisphere": "right",
        "region": "lateral prefrontal cortex",
        "radius_mm": 6,
    },
    # Add more ROIs as needed
}

# Preview what would be exported (dry-run mode)
print("=== DRY RUN ===")
output_files = export_timecourses_to_bids(
    csv_path=csv_path,
    bids_root=bids_root,
    network_label="cc200",  # Your network/pipeline label
    file_format="tsv",  # Options: "tsv", "csv", "npy"
    filename_prefix="timeseries",  # BIDS descriptor
    repetition_time=2.0,  # TR in seconds - UPDATE THIS!
    roi_metadata=roi_metadata,  # Optional ROI metadata
    task_label="rest",  # Optional task label (e.g., "rest", "nback")
    processing_description="Timeseries extracted after preprocessing with fMRIPrep",  # Optional
    dry_run=True,  # Preview without writing
)

# Now actually export to BIDS derivatives
print("\n=== ACTUAL EXPORT ===")
output_files = export_timecourses_to_bids(
    csv_path=csv_path,
    bids_root=bids_root,
    network_label="cc200",
    file_format="tsv",
    filename_prefix="timeseries",
    repetition_time=2.0,  # UPDATE THIS!
    roi_metadata=roi_metadata,
    task_label="rest",
    processing_description="Timeseries extracted after preprocessing with fMRIPrep",
    dry_run=False,  # Now actually write files
)

print(f"\nExported timecourses for {len(output_files)} subjects")
for subnum, files in output_files.items():
    print(f"  sub-{subnum}: {len(files)} files")


