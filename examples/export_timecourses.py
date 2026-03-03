"""
Example: Export timecourses to BIDS format

This example shows how to use fcmethods to parse timecourse data from a CSV
and export it to a BIDS-compliant derivative structure.
"""

from fcmethods import export_timecourses_to_bids

# Paths
csv_path = r"C:\path\to\timecourses.csv"
bids_root = r"C:\path\to\BIDS"

# Preview what would be exported (dry-run mode)
print("=== DRY RUN ===")
output_files = export_timecourses_to_bids(
    csv_path=csv_path,
    bids_root=bids_root,
    network_label="cc200",  # Change this to your network/parcellation label
    file_format="tsv",  # Options: "tsv", "csv", "npy"
    filename_prefix="timecourse",  # Customize the filename prefix
    dry_run=True,  # Preview without writing
)

# Now actually export to BIDS derivatives
print("\n=== ACTUAL EXPORT ===")
output_files = export_timecourses_to_bids(
    csv_path=csv_path,
    bids_root=bids_root,
    network_label="cc200",
    file_format="tsv",
    filename_prefix="timecourse",
    dry_run=False,  # Now actually write files
)

print(f"\nExported timecourses for {len(output_files)} subjects")
for subnum, files in output_files.items():
    print(f"  sub-{subnum}: {files}")

