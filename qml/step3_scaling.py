import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# --------------------------------------------------
# Load train/test data
# --------------------------------------------------

X_train = pd.read_csv("PQC_X_train.csv")
X_test = pd.read_csv("PQC_X_test.csv")

# --------------------------------------------------
# Create scaler
# --------------------------------------------------

scaler = StandardScaler()

# --------------------------------------------------
# FIT ONLY ON TRAINING DATA
# --------------------------------------------------

X_train_scaled = scaler.fit_transform(X_train)

# --------------------------------------------------
# Transform TEST DATA using the same scaler
# --------------------------------------------------

X_test_scaled = scaler.transform(X_test)

# --------------------------------------------------
# Convert back to DataFrames
# --------------------------------------------------

X_train_scaled = pd.DataFrame(
    X_train_scaled,
    columns=X_train.columns
)

X_test_scaled = pd.DataFrame(
    X_test_scaled,
    columns=X_test.columns
)

# --------------------------------------------------
# Check statistics
# --------------------------------------------------

print("=" * 60)
print("TRAINING DATA AFTER SCALING")
print("=" * 60)

print(
    X_train_scaled.describe().T[
        ["mean", "std", "min", "max"]
    ]
)

print("\n" + "=" * 60)
print("TEST DATA AFTER SCALING")
print("=" * 60)

print(
    X_test_scaled.describe().T[
        ["mean", "std", "min", "max"]
    ]
)

# --------------------------------------------------
# Save scaled datasets
# --------------------------------------------------

X_train_scaled.to_csv(
    "PQC_X_train_scaled.csv",
    index=False
)

X_test_scaled.to_csv(
    "PQC_X_test_scaled.csv",
    index=False
)

# --------------------------------------------------
# Save scaler
# --------------------------------------------------

joblib.dump(
    scaler,
    "PQC_scaler.pkl"
)

print("\nCreated:")
print("PQC_X_train_scaled.csv")
print("PQC_X_test_scaled.csv")
print("PQC_scaler.pkl")