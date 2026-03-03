"""
Example: Compute correlation matrices from BIDS timeseries

This example shows how to compute correlation matrices (with Fisher z-transform)
and difference matrices from timeseries data in a BIDS dataset.
"""

from fcmethods import compute_group_correlation_matrices

# Paths
BIDS_ROOT = "/path/to/BIDS"
NETWORK_LABEL = "cc200"  # Network label from your timeseries export
OUTPUT_ROOT = "/path/to/output/corrmat"

# Compute correlation matrices for all subjects
output_files = compute_group_correlation_matrices(
    bids_root=BIDS_ROOT,
    network_label=NETWORK_LABEL,
    output_root=OUTPUT_ROOT,
    intervention_label="rest-drug",  # Your intervention task label
    control_label="rest-placebo",  # Your control task label
    verbose=True,
)

print(f"\nProcessed {len(output_files)} subjects")
for sub_id, matrices in output_files.items():
    print(f"  sub-{sub_id}: {list(matrices.keys())}")
