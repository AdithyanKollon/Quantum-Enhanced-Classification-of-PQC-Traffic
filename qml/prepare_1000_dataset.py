"""
Prepare fixed 1,000-row datasets for the PQC classification experiments.

Experiment 1:
    Binary classification: PQC_label (0 = non-PQC, 1 = PQC)

Experiment 2:
    Six-class classification: Crypto_configuration for the six normal PQC
    configurations only.

All seven models in a given experiment must use the same saved train/test split.
The ID column is retained for traceability but must NOT be used as a model feature.
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TEST_SIZE = 0.20
SAMPLE_SIZE = 1000

MASTER_FILE = Path("PQC_master_dataset.csv")
OUTPUT_DIR = Path("experiment_data")
OUTPUT_DIR.mkdir(exist_ok=True)

FEATURES = [
    "e2c_total_bytes",
    "e4_entropy_h",
    "e5_entropy_c",
    "e6_time_char",
    "e6b_flow_duration_ms",
    "e2_client_size",
    "e2_client_record_len",
    "e2_server_record_len",
]

PQC_CONFIGURATIONS = [
    "ML-KEM-768 + RSA",
    "ML-KEM-768 + ML-DSA-65",
    "ML-KEM-768 + ML-DSA-87",
    "ML-KEM-1024 + RSA",
    "ML-KEM-1024 + ML-DSA-65",
    "ML-KEM-1024 + ML-DSA-87",
]


def save_experiment(data, target_col, prefix, stratify_col):
    """Save the selected 1,000 rows and common stratified train/test splits."""
    data = data.copy()
    if len(data) != SAMPLE_SIZE:
        raise ValueError(
            f"{prefix}: expected exactly {SAMPLE_SIZE} selected rows, got {len(data)}."
        )

    # Save the exact selected rows, including ID for traceability.
    data.to_csv(OUTPUT_DIR / f"{prefix}_1000.csv", index=False)

    # Split row indices first, then use those same indices for X and y.
    train_idx, test_idx = train_test_split(
        data.index,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=data[stratify_col],
    )

    X = data[FEATURES].copy()
    y = data[target_col].copy()

    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]

    # Save unscaled features. Fit scaling only on training data later.
    X_train.to_csv(OUTPUT_DIR / f"{prefix}_X_train.csv", index=False)
    X_test.to_csv(OUTPUT_DIR / f"{prefix}_X_test.csv", index=False)
    y_train.to_frame(name=target_col).to_csv(
        OUTPUT_DIR / f"{prefix}_y_train.csv", index=False
    )
    y_test.to_frame(name=target_col).to_csv(
        OUTPUT_DIR / f"{prefix}_y_test.csv", index=False
    )

    # Save row IDs separately so the exact split can be audited.
    data.loc[train_idx, ["ID", target_col]].to_csv(
        OUTPUT_DIR / f"{prefix}_train_rows.csv", index=False
    )
    data.loc[test_idx, ["ID", target_col]].to_csv(
        OUTPUT_DIR / f"{prefix}_test_rows.csv", index=False
    )

    print(f"\n{'=' * 65}")
    print(f"{prefix.upper()}")
    print(f"{'=' * 65}")
    print("Selected rows:", len(data))
    print("Training rows:", len(X_train))
    print("Testing rows:", len(X_test))
    print("\nFull selected target distribution:")
    print(data[target_col].value_counts().sort_index().to_string())
    print("\nTraining target distribution:")
    print(y_train.value_counts().sort_index().to_string())
    print("\nTesting target distribution:")
    print(y_test.value_counts().sort_index().to_string())
    print("Train/test ID overlap:",
          len(set(data.loc[train_idx, "ID"]) & set(data.loc[test_idx, "ID"])))
    print("Saved files under:", OUTPUT_DIR.resolve())


def main():
    if not MASTER_FILE.exists():
        raise FileNotFoundError(
            f"Cannot find {MASTER_FILE}. Run this script from the folder "
            "containing PQC_master_dataset.csv."
        )

    df = pd.read_csv(MASTER_FILE)

    required_cols = FEATURES + [
        "PQC_label", "Crypto_configuration", "ID"
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if df[required_cols].isnull().any().any():
        null_counts = df[required_cols].isnull().sum()
        null_counts = null_counts[null_counts > 0]
        raise ValueError(
            "Missing values found in required columns:\n"
            + null_counts.to_string()
        )

    if len(df) < SAMPLE_SIZE:
        raise ValueError(f"Master dataset has fewer than {SAMPLE_SIZE} rows.")

    # ------------------------------------------------------------
    # Experiment 1: binary PQC vs non-PQC
    # Preserve the original class proportions through stratification.
    # ------------------------------------------------------------
    binary_df = df.copy()
    binary_df["PQC_label"] = binary_df["PQC_label"].astype(int)

    # Stratified sampling to exactly 1,000 rows.
    binary_sample, _ = train_test_split(
        binary_df,
        train_size=SAMPLE_SIZE,
        random_state=RANDOM_STATE,
        stratify=binary_df["PQC_label"],
    )
    save_experiment(
        binary_sample,
        target_col="PQC_label",
        prefix="binary",
        stratify_col="PQC_label",
    )

    # ------------------------------------------------------------
    # Experiment 2: six normal PQC configurations
    # Exclude non-PQC configurations and anomaly/network-condition
    # categories because this experiment targets six crypto configs.
    # ------------------------------------------------------------
    multiclass_df = df[
        df["Crypto_configuration"].isin(PQC_CONFIGURATIONS)
    ].copy()

    counts = multiclass_df["Crypto_configuration"].value_counts()
    absent = [c for c in PQC_CONFIGURATIONS if c not in counts.index]
    if absent:
        raise ValueError(f"Expected PQC configurations missing: {absent}")

    if (counts < 1).any() or len(multiclass_df) < SAMPLE_SIZE:
        raise ValueError(
            "Not enough rows in the six PQC configurations for a 1,000-row sample."
        )

    multiclass_sample, _ = train_test_split(
        multiclass_df,
        train_size=SAMPLE_SIZE,
        random_state=RANDOM_STATE,
        stratify=multiclass_df["Crypto_configuration"],
    )
    save_experiment(
        multiclass_sample,
        target_col="Crypto_configuration",
        prefix="multiclass",
        stratify_col="Crypto_configuration",
    )

    print("\nPreparation complete.")
    print("Use these saved splits for all seven models within each experiment.")
    print("Do not fit scalers or feature selection on test data.")


if __name__ == "__main__":
    main()
