import pandas as pd
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "ml_features_and_labels.csv"
OUTPUT_FILE = "PQC_master_dataset.csv"

# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("=" * 70)
print("RAW DATASET")
print("=" * 70)
print("Shape:", df.shape)


# ============================================================
# 2. EXTRACT EXPERIMENT NAME
# ============================================================

# Example:
# normal_mlkem768_mldsa65_cert_run123
#
# becomes:
# normal_mlkem768_mldsa65_cert

df["experiment"] = (
    df["ID"]
    .astype(str)
    .str.replace(r"_run\d+$", "", regex=True)
)

print("\n" + "=" * 70)
print("EXPERIMENT CONFIGURATIONS")
print("=" * 70)

print(
    df["experiment"]
    .value_counts()
    .to_string()
)


# ============================================================
# 3. CLASSICAL / NON-PQC EXPERIMENTS
# ============================================================

classical_experiments = {

    "normal_X25519_ecdsa_p384_cert",
    "normal_X25519_rsa3072_cert",
    "normal_X25519_rsa_cert",
    "normal_X25519_ecdsa_p256_cert",

    "net_high_jitter_classic",

    "threat_data_exfil",
    "model_gray_rsa_mimicry",

    "raf_alg_mismatch",
    "raf_downgrade_overt",
    "raf_robustness_malformed_keyshare"
}


# ============================================================
# 4. PQC EXPERIMENTS
# ============================================================

pqc_experiments = {

    "normal_mlkem1024_mldsa65_cert",
    "normal_mlkem1024_mldsa87_cert",
    "normal_mlkem1024_rsa_cert",

    "normal_mlkem768_mldsa65_cert",
    "normal_mlkem768_mldsa87_cert",
    "normal_mlkem768_rsa_cert",

    "net_high_loss_pqc",
    "net_high_latency_pqc",

    "anomaly_hybrid_pki",
    "anomaly_kem_mismatch",
    "anomaly_pqc_over_tls12"
}


# ============================================================
# 5. CHECK THAT EVERY EXPERIMENT IS ACCOUNTED FOR
# ============================================================

known_experiments = (
    classical_experiments |
    pqc_experiments
)

actual_experiments = set(
    df["experiment"].unique()
)

unrecognized = actual_experiments - known_experiments

print("\n" + "=" * 70)
print("EXPERIMENT MAPPING CHECK")
print("=" * 70)

if unrecognized:

    print("WARNING - Unrecognized experiments:")
    for x in sorted(unrecognized):
        print("  ", x)

    raise ValueError(
        "Some experiments have not been classified. "
        "Do not continue until they are resolved."
    )

else:

    print("PASS: All experiments are classified.")


# ============================================================
# 6. BINARY PQC LABEL
# ============================================================

# 0 = Non-PQC
# 1 = PQC

df["PQC_label"] = np.nan

df.loc[
    df["experiment"].isin(classical_experiments),
    "PQC_label"
] = 0

df.loc[
    df["experiment"].isin(pqc_experiments),
    "PQC_label"
] = 1


# ============================================================
# 7. EXTRACT PQC KEM
# ============================================================

def get_pqc_kem(experiment):

    experiment = experiment.lower()

    if "mlkem768" in experiment:
        return "ML-KEM-768"

    if "mlkem1024" in experiment:
        return "ML-KEM-1024"

    # The experiment is PQC-related but doesn't explicitly
    # identify ML-KEM-768 or ML-KEM-1024.
    return "None"


df["PQC_KEM"] = df["experiment"].apply(get_pqc_kem)


# ============================================================
# 8. EXTRACT PQC SIGNATURE
# ============================================================

def get_pqc_signature(experiment):

    experiment = experiment.lower()

    if "mldsa65" in experiment:
        return "ML-DSA-65"

    if "mldsa87" in experiment:
        return "ML-DSA-87"

    return "None"


df["PQC_signature"] = df["experiment"].apply(
    get_pqc_signature
)


# ============================================================
# 9. CLASSIFY PQC CONFIGURATION
# ============================================================

def get_pqc_class(row):

    experiment = row["experiment"].lower()

    # --------------------------------------------------------
    # Classical
    # --------------------------------------------------------

    if row["PQC_label"] == 0:
        return "Non-PQC"

    # --------------------------------------------------------
    # PQC anomaly experiments
    # --------------------------------------------------------

    if experiment.startswith("anomaly_"):
        return "PQC-Anomalous"

    # --------------------------------------------------------
    # Normal PQC configurations
    # --------------------------------------------------------

    # ML-KEM + RSA = hybrid
    if "rsa" in experiment and "mlkem" in experiment:
        return "Hybrid-PQC"

    # ML-KEM + ML-DSA = pure PQC
    if "mlkem" in experiment and "mldsa" in experiment:
        return "Pure-PQC"

    # Network PQC experiments
    if "pqc" in experiment:
        return "PQC-Anomalous"

    return "PQC"


df["PQC_class"] = df.apply(
    get_pqc_class,
    axis=1
)


# ============================================================
# 10. PRESERVE EXACT CRYPTO CONFIGURATION
# ============================================================

def get_crypto_configuration(experiment):

    experiment = experiment.lower()

    # ML-KEM + ML-DSA
    if "mlkem768" in experiment and "mldsa65" in experiment:
        return "ML-KEM-768 + ML-DSA-65"

    if "mlkem768" in experiment and "mldsa87" in experiment:
        return "ML-KEM-768 + ML-DSA-87"

    if "mlkem1024" in experiment and "mldsa65" in experiment:
        return "ML-KEM-1024 + ML-DSA-65"

    if "mlkem1024" in experiment and "mldsa87" in experiment:
        return "ML-KEM-1024 + ML-DSA-87"

    # ML-KEM + RSA
    if "mlkem768" in experiment and "rsa" in experiment:
        return "ML-KEM-768 + RSA"

    if "mlkem1024" in experiment and "rsa" in experiment:
        return "ML-KEM-1024 + RSA"

    # Classical configurations
    if "x25519" in experiment:
        if "ecdsa_p256" in experiment:
            return "X25519 + ECDSA-P256"

        if "ecdsa_p384" in experiment:
            return "X25519 + ECDSA-P384"

        if "rsa3072" in experiment:
            return "X25519 + RSA-3072"

        if "_rsa" in experiment:
            return "X25519 + RSA"

    # Network / anomaly configurations
    return experiment


df["Crypto_configuration"] = df["experiment"].apply(
    get_crypto_configuration
)


# ============================================================
# 11. TRAFFIC FEATURES
# ============================================================

traffic_features = [

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
# 12. CHECK FEATURES
# ============================================================

missing_features = [
    feature
    for feature in traffic_features
    if feature not in df.columns
]

if missing_features:

    raise ValueError(
        f"Missing features: {missing_features}"
    )


# ============================================================
# 13. HANDLE INF VALUES
# ============================================================

df[traffic_features] = df[
    traffic_features
].replace(
    [np.inf, -np.inf],
    np.nan
)


# ============================================================
# 14. CREATE MASTER DATASET
# ============================================================

master_columns = [

    # --------------------------------------------------------
    # Traffic features
    # --------------------------------------------------------

    "e2c_total_bytes",
    "e4_entropy_h",
    "e5_entropy_c",
    "e6_time_char",
    "e6b_flow_duration_ms",
    "e2_client_size",
    "e2_client_record_len",
    "e2_server_record_len",

    # --------------------------------------------------------
    # Main target
    # --------------------------------------------------------

    "PQC_label",

    # --------------------------------------------------------
    # PQC categorisation
    # --------------------------------------------------------

    "PQC_class",
    "PQC_KEM",
    "PQC_signature",

    # --------------------------------------------------------
    # Exact configuration
    # --------------------------------------------------------

    "Crypto_configuration",

    # --------------------------------------------------------
    # Experiment information
    # --------------------------------------------------------

    "experiment",
    "ID"
]


master_df = df[master_columns].copy()


# ============================================================
# 15. REMOVE MISSING FEATURE ROWS
# ============================================================

before = len(master_df)

master_df = master_df.dropna(
    subset=traffic_features
).reset_index(drop=True)

after = len(master_df)

print("\n" + "=" * 70)
print("MISSING DATA")
print("=" * 70)

print("Before:", before)
print("After :", after)
print("Removed:", before - after)


# ============================================================
# 16. BINARY PQC DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("BINARY PQC DISTRIBUTION")
print("=" * 70)

print(
    master_df["PQC_label"]
    .value_counts()
    .sort_index()
)


print("\nReadable:")

print(
    master_df["PQC_label"]
    .map({
        0: "Non-PQC",
        1: "PQC"
    })
    .value_counts()
)


# ============================================================
# 17. PQC CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("PQC CLASS DISTRIBUTION")
print("=" * 70)

print(
    master_df["PQC_class"]
    .value_counts()
)


# ============================================================
# 18. PQC KEM DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("PQC KEM DISTRIBUTION")
print("=" * 70)

print(
    master_df["PQC_KEM"]
    .value_counts()
)


# ============================================================
# 19. PQC SIGNATURE DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("PQC SIGNATURE DISTRIBUTION")
print("=" * 70)

print(
    master_df["PQC_signature"]
    .value_counts()
)


# ============================================================
# 20. CRYPTO CONFIGURATION DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("CRYPTO CONFIGURATION DISTRIBUTION")
print("=" * 70)

print(
    master_df["Crypto_configuration"]
    .value_counts()
)


# ============================================================
# 21. EXPERIMENT MAPPING
# ============================================================

print("\n" + "=" * 70)
print("FULL EXPERIMENT MAPPING")
print("=" * 70)

mapping = (
    master_df
    .groupby(
        [
            "experiment",
            "PQC_label",
            "PQC_class",
            "PQC_KEM",
            "PQC_signature",
            "Crypto_configuration"
        ]
    )
    .size()
    .reset_index(name="samples")
)

print(
    mapping.to_string(index=False)
)


# ============================================================
# 22. CHECK LABEL CONSISTENCY
# ============================================================

print("\n" + "=" * 70)
print("CONSISTENCY CHECK")
print("=" * 70)

label_consistency = (
    master_df
    .groupby("experiment")["PQC_label"]
    .nunique()
)

bad_labels = label_consistency[
    label_consistency > 1
]

if len(bad_labels) == 0:

    print(
        "PASS: Every experiment has one consistent PQC label."
    )

else:

    print(
        "WARNING: Inconsistent PQC labels found:"
    )

    print(bad_labels)


# ============================================================
# 23. SAVE MASTER DATASET
# ============================================================

master_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("MASTER DATASET CREATED")
print("=" * 70)

print("File:", OUTPUT_FILE)
print("Shape:", master_df.shape)

print("\nColumns:")

for column in master_df.columns:
    print(" -", column)


# ============================================================
# 24. CREATE BINARY CLASSIFICATION DATASET
# ============================================================

binary_df = master_df[
    [
        *traffic_features,
        "PQC_label"
    ]
].copy()

binary_df.to_csv(
    "PQC_binary_dataset.csv",
    index=False
)


# ============================================================
# 25. CREATE PQC-ONLY DATASET
# ============================================================

pqc_df = master_df[
    master_df["PQC_label"] == 1
].copy()

pqc_df.to_csv(
    "PQC_only_dataset.csv",
    index=False
)


# ============================================================
# 26. CREATE PQC KEM CLASSIFICATION DATASET
# ============================================================
#
# Only samples with a known ML-KEM type.
#
# Target:
# ML-KEM-768
# ML-KEM-1024
# ============================================================

kem_df = master_df[
    master_df["PQC_KEM"].isin(
        [
            "ML-KEM-768",
            "ML-KEM-1024"
        ]
    )
].copy()

kem_columns = [
    *traffic_features,
    "PQC_KEM",
    "PQC_signature",
    "Crypto_configuration"
]

kem_df[kem_columns].to_csv(
    "PQC_KEM_dataset.csv",
    index=False
)


# ============================================================
# 27. FINAL FILE SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FILES CREATED")
print("=" * 70)

print("1. PQC_master_dataset.csv")
print("   Complete master dataset")

print("2. PQC_binary_dataset.csv")
print("   Non-PQC vs PQC")

print("3. PQC_only_dataset.csv")
print("   PQC traffic only")

print("4. PQC_KEM_dataset.csv")
print("   ML-KEM-768 vs ML-KEM-1024")

print("\nDone.")