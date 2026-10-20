import pandas as pd
import numpy as np

# Load master dataset
df = pd.read_csv("PQC_master_dataset.csv")

print("=" * 60)
print("DATASET VALIDATION")
print("=" * 60)

# 1. Shape
print("\n1. Dataset shape:")
print(df.shape)

# 2. Columns
print("\n2. Columns:")
print(df.columns.tolist())

# 3. Missing values
print("\n3. Missing values:")
print(df.isnull().sum())

# 4. Duplicate rows
print("\n4. Duplicate rows:")
print(df.duplicated().sum())

# 5. Binary PQC distribution
print("\n5. PQC distribution:")
print(df["PQC_label"].value_counts())

print("\nPQC percentages:")
print(
    df["PQC_label"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

# 6. PQC class distribution
print("\n6. PQC class distribution:")
print(df["PQC_class"].value_counts())

# 7. PQC KEM distribution
print("\n7. PQC KEM distribution:")
print(df["PQC_KEM"].value_counts())

# 8. PQC signature distribution
print("\n8. PQC signature distribution:")
print(df["PQC_signature"].value_counts())

# 9. Crypto configuration
print("\n9. Crypto configuration:")
print(df["Crypto_configuration"].value_counts())

# 10. Numerical feature summary
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

print("\n10. Feature statistics:")
print(df[features].describe().T)

# 11. Check infinite values
print("\n11. Infinite values:")
print(np.isinf(df[features]).sum())

# 12. Check target validity
print("\n12. Invalid PQC labels:")
print(
    df.loc[
        ~df["PQC_label"].isin([0, 1]),
        "PQC_label"
    ].value_counts()
)

print("\n" + "=" * 60)
print("VALIDATION COMPLETE")
print("=" * 60)