import pandas as pd

# Load master dataset
df = pd.read_csv("PQC_master_dataset.csv")

# --------------------------------------------------
# Features
# --------------------------------------------------

features = [
    "e2c_total_bytes",
    "e4_entropy_h",
    "e5_entropy_c",
    "e6_time_char",
    "e6b_flow_duration_ms",
    "e2_client_size",
    "e2_client_record_len",
    "e2_server_record_len"
]

# --------------------------------------------------
# Create X and y
# --------------------------------------------------

X = df[features].copy()
y = df["PQC_label"].astype(int).copy()

# --------------------------------------------------
# Check
# --------------------------------------------------

print("X shape:", X.shape)
print("y shape:", y.shape)

print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget distribution:")
print(y.value_counts())

print("\nTarget percentages:")
print(
    y.value_counts(normalize=True)
    .mul(100)
    .round(2)
)

# --------------------------------------------------
# Save
# --------------------------------------------------

X.to_csv("PQC_X_features.csv", index=False)

y.to_csv(
    "PQC_y_target.csv",
    index=False,
    header=["PQC_label"]
)

print("\nCreated:")
print("PQC_X_features.csv")
print("PQC_y_target.csv")