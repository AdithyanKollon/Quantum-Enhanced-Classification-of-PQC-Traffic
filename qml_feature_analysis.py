import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_selection import mutual_info_classif

DATA_DIR = "experiment_data"
OUTPUT_DIR = os.path.join(DATA_DIR, "qml_feature_analysis")

os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURES = [
    "e2c_total_bytes",
    "e4_entropy_h",
    "e5_entropy_c",
    "e6_time_char",
    "e6b_flow_duration_ms",
    "e2_client_size",
    "e2_client_record_len",
    "e2_server_record_len"
]


def analyze_features(experiment_name, target_name):
    """
    Calculate Mutual Information using training data only.
    """

    x_path = os.path.join(
        DATA_DIR, f"{experiment_name}_X_train.csv"
    )

    y_path = os.path.join(
        DATA_DIR, f"{experiment_name}_y_train.csv"
    )

    X = pd.read_csv(x_path)
    y = pd.read_csv(y_path).iloc[:, 0]

    missing_features = [
        feature for feature in FEATURES
        if feature not in X.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing features in {x_path}: {missing_features}"
        )

    X = X[FEATURES]

    if X.isnull().any().any():
        raise ValueError("X contains missing values.")

    scores = mutual_info_classif(
        X,
        y,
        random_state=42
    )

    results = pd.DataFrame({
        "Feature": FEATURES,
        "Mutual_Information": scores
    }).sort_values(
        "Mutual_Information",
        ascending=False
    )

    results.to_csv(
        os.path.join(
            OUTPUT_DIR,
            f"{experiment_name}_mutual_information.csv"
        ),
        index=False
    )

    plt.figure(figsize=(10, 6))

    plt.barh(
        results["Feature"][::-1],
        results["Mutual_Information"][::-1]
    )

    plt.xlabel("Mutual Information Score")
    plt.ylabel("Feature")

    plt.title(
        f"Mutual Information Feature Ranking: {target_name}"
    )

    plt.tight_layout()

    plot_path = os.path.join(
        OUTPUT_DIR,
        f"{experiment_name}_feature_ranking.png"
    )

    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"\n{target_name} feature ranking:")
    print(results.to_string(index=False))
    print(f"Plot saved to: {plot_path}")


# Binary PQC detection
analyze_features(
    experiment_name="binary",
    target_name="Binary PQC Detection"
)

# Six-class cryptographic configuration classification
analyze_features(
    experiment_name="multiclass",
    target_name="Multiclass Crypto Configuration"
)

print("\nFeature analysis completed.")