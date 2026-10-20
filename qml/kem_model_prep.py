import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# ============================================================
# LOAD MASTER DATASET
# ============================================================

df = pd.read_csv("PQC_master_dataset.csv")

print("Original dataset shape:", df.shape)


# ============================================================
# TRAFFIC FEATURES
# ============================================================

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


# ============================================================
# KEEP ONLY ML-KEM TRAFFIC
# ============================================================

kem_values = [
    "ML-KEM-768",
    "ML-KEM-1024"
]

df_kem = df[df["PQC_KEM"].isin(kem_values)].copy()

print("\nKEM dataset shape:", df_kem.shape)


# ============================================================
# CREATE TARGET
# ============================================================

# 0 = ML-KEM-768
# 1 = ML-KEM-1024

df_kem["KEM_label"] = df_kem["PQC_KEM"].map({
    "ML-KEM-768": 0,
    "ML-KEM-1024": 1
})


# ============================================================
# REMOVE MISSING VALUES
# ============================================================

df_kem = df_kem.dropna(subset=features)


# ============================================================
# CREATE X AND y
# ============================================================

X = df_kem[features].copy()
y = df_kem["KEM_label"].copy()


# ============================================================
# CHECK DISTRIBUTION
# ============================================================

print("\nKEM class distribution:")
print(df_kem["PQC_KEM"].value_counts())

print("\nTarget distribution:")
print(y.value_counts())

print("\nFeatures used:")
print(features)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n" + "=" * 60)
print("TRAIN / TEST SPLIT")
print("=" * 60)

print("\nTraining shape:", X_train.shape)
print("Testing shape :", X_test.shape)

print("\nTraining distribution:")
print(y_train.value_counts())

print("\nTesting distribution:")
print(y_test.value_counts())


# ============================================================
# SAVE UNSCALED DATA
# ============================================================

X_train.to_csv(
    "KEM_X_train.csv",
    index=False
)

X_test.to_csv(
    "KEM_X_test.csv",
    index=False
)

y_train.to_csv(
    "KEM_y_train.csv",
    index=False,
    header=["KEM_label"]
)

y_test.to_csv(
    "KEM_y_test.csv",
    index=False,
    header=["KEM_label"]
)


# ============================================================
# FEATURE SCALING
# ============================================================

scaler = StandardScaler()

# IMPORTANT:
# Fit scaler ONLY on training data
X_train_scaled = scaler.fit_transform(X_train)

# Apply the SAME scaler to test data
X_test_scaled = scaler.transform(X_test)


# Convert back to DataFrames
X_train_scaled = pd.DataFrame(
    X_train_scaled,
    columns=features
)

X_test_scaled = pd.DataFrame(
    X_test_scaled,
    columns=features
)


# ============================================================
# SAVE SCALED DATA
# ============================================================

X_train_scaled.to_csv(
    "KEM_X_train_scaled.csv",
    index=False
)

X_test_scaled.to_csv(
    "KEM_X_test_scaled.csv",
    index=False
)

joblib.dump(
    scaler,
    "KEM_scaler.pkl"
)


# ============================================================
# VERIFY SCALING
# ============================================================

print("\n" + "=" * 60)
print("SCALING CHECK")
print("=" * 60)

print("\nTraining means:")
print(X_train_scaled.mean().round(4))

print("\nTraining standard deviations:")
print(X_train_scaled.std().round(4))


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("KEM DATA PREPARATION COMPLETE")
print("=" * 60)

print("""
Created:

KEM_X_features.csv
KEM_y_target.csv

KEM_X_train.csv
KEM_X_test.csv
KEM_y_train.csv
KEM_y_test.csv

KEM_X_train_scaled.csv
KEM_X_test_scaled.csv
KEM_scaler.pkl
""")