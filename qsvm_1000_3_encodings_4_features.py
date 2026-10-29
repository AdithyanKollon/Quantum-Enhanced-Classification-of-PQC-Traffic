import os
import time
import warnings

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel

from sklearn.preprocessing import MinMaxScaler, LabelEncoder, label_binarize
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================
DATA_DIR = "experiment_data"
OUTPUT_DIR = os.path.join(DATA_DIR, "qsvm_3_encoding_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# These are the same four features retained in your uploaded scripts
# after removing the four highest-ranked features.
FEATURES = [
    "e4_entropy_h",
    "e5_entropy_c",
    "e6_time_char",
    "e6b_flow_duration_ms",
]

EXPERIMENTS = {
    "binary": {
        "target_name": "PQC_label",
        "class_names": ["Non-PQC", "PQC"],
    },
    "multiclass": {
        "target_name": "Crypto_configuration",
        "class_names": None,
    },
}

RANDOM_STATE = 42


def amplitude_kernel(A, B):
    """Exact squared-overlap kernel for normalized four-value states."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    A_norms = np.linalg.norm(A, axis=1, keepdims=True)
    B_norms = np.linalg.norm(B, axis=1, keepdims=True)
    if np.any(A_norms == 0) or np.any(B_norms == 0):
        raise ValueError("Zero vector found during amplitude normalization.")
    A = A / A_norms
    B = B / B_norms
    return np.square(A @ B.T)


def make_angle_map():
    x = ParameterVector("x", 4)
    qc = QuantumCircuit(4)
    for i in range(4):
        qc.ry(x[i], i)
    for i in range(3):
        qc.cz(i, i + 1)
    return qc


def prepare_data(experiment):
    x_train_path = os.path.join(DATA_DIR, f"{experiment}_X_train.csv")
    x_test_path = os.path.join(DATA_DIR, f"{experiment}_X_test.csv")
    y_train_path = os.path.join(DATA_DIR, f"{experiment}_y_train.csv")
    y_test_path = os.path.join(DATA_DIR, f"{experiment}_y_test.csv")

    X_train_df = pd.read_csv(x_train_path)
    X_test_df = pd.read_csv(x_test_path)
    y_train_raw = pd.read_csv(y_train_path).iloc[:, 0]
    y_test_raw = pd.read_csv(y_test_path).iloc[:, 0]

    missing = [f for f in FEATURES if f not in X_train_df.columns or f not in X_test_df.columns]
    if missing:
        raise ValueError(
            f"{experiment}: missing required features {missing}. "
            "Check that the experiment CSVs contain the same feature names."
        )

    X_train_df = X_train_df[FEATURES].astype(float)
    X_test_df = X_test_df[FEATURES].astype(float)

    # Fit scaling on training data only, then map values into rotation angles.
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_train = scaler.fit_transform(X_train_df)
    X_test = scaler.transform(X_test_df)
    X_test = np.clip(X_test, 0, np.pi)

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train_raw.astype(str))
    try:
        y_test = encoder.transform(y_test_raw.astype(str))
    except ValueError as exc:
        raise ValueError(f"{experiment}: test set contains a label absent from training data.") from exc

    return X_train, X_test, y_train, y_test, encoder


def evaluate_experiment(experiment, config):
    X_train, X_test, y_train, y_test, encoder = prepare_data(experiment)
    classes = np.unique(y_train)
    multiclass = len(classes) > 2

    print("\n" + "=" * 78)
    print(f"QSVM EXPERIMENT: {experiment.upper()}")
    print(f"Train samples: {len(y_train)} | Test samples: {len(y_test)}")
    print(f"Features used: {FEATURES}")
    print(f"Classes: {list(encoder.classes_)}")
    print("=" * 78)

    angle_kernel = FidelityQuantumKernel(feature_map=make_angle_map())
    zz_map = ZZFeatureMap(feature_dimension=4, reps=2, entanglement="linear")
    zz_kernel = FidelityQuantumKernel(feature_map=zz_map)

    results = []
    plot_data = {}

    encodings = [
        ("Angle Encoding", angle_kernel),
        ("Amplitude Encoding", None),
        ("ZZFeatureMap", zz_kernel),
    ]

    for name, qkernel in encodings:
        print(f"\nRunning {name}...")
        kernel_start = time.time()

        if name == "Amplitude Encoding":
            K_train = amplitude_kernel(X_train, X_train)
            K_test = amplitude_kernel(X_test, X_train)
        else:
            K_train = qkernel.evaluate(x_vec=X_train)
            K_test = qkernel.evaluate(x_vec=X_test, y_vec=X_train)

        kernel_time = time.time() - kernel_start

        model = SVC(
            kernel="precomputed",
            decision_function_shape="ovr",
            probability=True,
            random_state=RANDOM_STATE
        )

        train_start = time.time()
        model.fit(K_train, y_train)
        training_time = time.time() - train_start

        predict_start = time.time()
        y_pred = model.predict(K_test)
        y_prob = model.predict_proba(K_test)
        inference_time = time.time() - predict_start

        accuracy = accuracy_score(y_test, y_pred)
        average = "macro" if multiclass else "binary"
        precision = precision_score(y_test, y_pred, average=average, zero_division=0)
        recall = recall_score(y_test, y_pred, average=average, zero_division=0)
        f1 = f1_score(y_test, y_pred, average=average, zero_division=0)

        try:
            if multiclass:
                y_true_bin = label_binarize(y_test, classes=classes)
                roc_auc = roc_auc_score(y_true_bin, y_prob, multi_class="ovr", average="macro")
            else:
                roc_auc = roc_auc_score(y_test, y_prob[:, 1])
        except (ValueError, IndexError):
            roc_auc = np.nan

        cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(encoder.classes_)))
        print(f"Accuracy       : {accuracy:.4f}")
        print(f"Precision      : {precision:.4f}")
        print(f"Recall         : {recall:.4f}")
        print(f"F1-score       : {f1:.4f}")
        print(f"ROC-AUC        : {roc_auc:.4f}" if np.isfinite(roc_auc) else "ROC-AUC        : unavailable")
        print(f"Kernel time    : {kernel_time:.2f} s")
        print(f"Training time  : {training_time:.2f} s")
        print(f"Inference time : {inference_time:.2f} s")
        print("Confusion matrix:\n", cm)

        results.append({
            "Experiment": experiment,
            "Model": "QSVM",
            "Encoding": name,
            "Training samples": len(y_train),
            "Testing samples": len(y_test),
            "Features": len(FEATURES),
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "ROC_AUC": roc_auc,
            "Kernel time (s)": kernel_time,
            "Training time (s)": training_time,
            "Inference time (s)": inference_time,
        })

        fig, ax = plt.subplots(figsize=(7, 6))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=encoder.classes_
        )
        disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
        ax.set_title(f"QSVM {name}: {experiment.capitalize()}")
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, f"{experiment}_{name.lower().replace(' ', '_')}_confusion_matrix.png"), dpi=250)
        plt.close(fig)

        plot_data[name] = accuracy

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(OUTPUT_DIR, f"{experiment}_qsvm_results.csv"), index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(list(plot_data.keys()), list(plot_data.values()))
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.set_title(f"QSVM Encoding Comparison: {experiment.capitalize()}")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f"{experiment}_qsvm_encoding_comparison.png"), dpi=250)
    plt.close(fig)

    return results_df


if __name__ == "__main__":
    all_results = []
    for exp_name, exp_config in EXPERIMENTS.items():
        all_results.append(evaluate_experiment(exp_name, exp_config))

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(os.path.join(OUTPUT_DIR, "qsvm_all_results.csv"), index=False)
    print("\nFINAL QSVM COMPARISON")
    print(combined.to_string(index=False))
    print(f"\nAll results and plots saved in: {OUTPUT_DIR}")
    print("Note: exact quantum-kernel evaluation on 800 training samples can take a long time.")
