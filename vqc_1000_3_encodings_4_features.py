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
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.algorithms import VQC
from qiskit_algorithms.optimizers import COBYLA

from sklearn.preprocessing import MinMaxScaler, LabelEncoder, label_binarize
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================
DATA_DIR = "experiment_data"
OUTPUT_DIR = os.path.join(DATA_DIR, "vqc_3_encoding_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Same four least-informative features retained in the uploaded script.
FEATURES = [
    "e4_entropy_h",
    "e5_entropy_c",
    "e6_time_char",
    "e6b_flow_duration_ms",
]

EXPERIMENTS = ["binary", "multiclass"]
RANDOM_STATE = 42
MAX_ITER = 30


def create_angle_feature_map(n=4):
    x = ParameterVector("x", n)
    qc = QuantumCircuit(n)
    for i in range(n):
        qc.ry(x[i], i)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    return qc


def create_amplitude_inspired_map():
    # Preserves the encoding approach from the uploaded script.
    # This is an amplitude-inspired data encoding, not arbitrary
    # amplitude encoding of every input vector.
    x = ParameterVector("x", 4)
    qc = QuantumCircuit(2)
    qc.initialize([0.5, 0.5, 0.5, 0.5], [0, 1])
    qc.ry(x[0], 0)
    qc.ry(x[1], 1)
    qc.cx(0, 1)
    qc.ry(x[2], 0)
    qc.ry(x[3], 1)
    return qc


def prepare_data(experiment):
    paths = {
        "x_train": os.path.join(DATA_DIR, f"{experiment}_X_train.csv"),
        "x_test": os.path.join(DATA_DIR, f"{experiment}_X_test.csv"),
        "y_train": os.path.join(DATA_DIR, f"{experiment}_y_train.csv"),
        "y_test": os.path.join(DATA_DIR, f"{experiment}_y_test.csv"),
    }
    X_train_df = pd.read_csv(paths["x_train"])
    X_test_df = pd.read_csv(paths["x_test"])
    y_train_raw = pd.read_csv(paths["y_train"]).iloc[:, 0]
    y_test_raw = pd.read_csv(paths["y_test"]).iloc[:, 0]

    missing = [f for f in FEATURES if f not in X_train_df.columns or f not in X_test_df.columns]
    if missing:
        raise ValueError(
            f"{experiment}: missing required features {missing}. "
            "Check the experiment CSV columns."
        )

    X_train_df = X_train_df[FEATURES].astype(float)
    X_test_df = X_test_df[FEATURES].astype(float)

    # Fit preprocessing only on the training data.
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_train = scaler.fit_transform(X_train_df)
    X_test = np.clip(scaler.transform(X_test_df), 0, np.pi)

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train_raw.astype(str))
    y_test = encoder.transform(y_test_raw.astype(str))

    return X_train, X_test, y_train, y_test, encoder


def evaluate_experiment(experiment):
    X_train, X_test, y_train, y_test, encoder = prepare_data(experiment)
    multiclass = len(encoder.classes_) > 2

    print("\n" + "=" * 78)
    print(f"VQC EXPERIMENT: {experiment.upper()}")
    print(f"Train samples: {len(y_train)} | Test samples: {len(y_test)}")
    print(f"Features used: {FEATURES}")
    print(f"Classes: {list(encoder.classes_)}")
    print("=" * 78)

    feature_maps = {
        "Angle Encoding": create_angle_feature_map(4),
        "Amplitude Encoding": create_amplitude_inspired_map(),
        "ZZFeatureMap": ZZFeatureMap(
            feature_dimension=4,
            reps=2,
            entanglement="linear"
        ),
    }

    results = []
    accuracy_by_encoding = {}

    for name, feature_map in feature_maps.items():
        print(f"\nRunning VQC with {name}...")
        num_qubits = feature_map.num_qubits
        ansatz = RealAmplitudes(num_qubits=num_qubits, reps=1)
        optimizer = COBYLA(maxiter=MAX_ITER)

        model = VQC(
            feature_map=feature_map,
            ansatz=ansatz,
            optimizer=optimizer
        )

        start = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start

        start = time.time()
        y_pred = np.asarray(model.predict(X_test)).reshape(-1).astype(int)
        probabilities = np.asarray(model.predict_proba(X_test))
        inference_time = time.time() - start

        # Normalize probability output shape and class order.
        if probabilities.ndim != 2 or probabilities.shape[1] != len(encoder.classes_):
            raise ValueError(
                f"{name}: expected probability shape "
                f"(n_test, {len(encoder.classes_)}), got {probabilities.shape}"
            )

        accuracy = accuracy_score(y_test, y_pred)
        average = "macro" if multiclass else "binary"
        precision = precision_score(y_test, y_pred, average=average, zero_division=0)
        recall = recall_score(y_test, y_pred, average=average, zero_division=0)
        f1 = f1_score(y_test, y_pred, average=average, zero_division=0)

        try:
            if multiclass:
                y_true_bin = label_binarize(y_test, classes=np.arange(len(encoder.classes_)))
                roc_auc = roc_auc_score(
                    y_true_bin, probabilities, multi_class="ovr", average="macro"
                )
            else:
                roc_auc = roc_auc_score(y_test, probabilities[:, 1])
        except (ValueError, IndexError):
            roc_auc = np.nan

        cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(encoder.classes_)))

        print(f"Qubits         : {num_qubits}")
        print(f"Accuracy       : {accuracy:.4f}")
        print(f"Precision      : {precision:.4f}")
        print(f"Recall         : {recall:.4f}")
        print(f"F1-score       : {f1:.4f}")
        print(f"ROC-AUC        : {roc_auc:.4f}" if np.isfinite(roc_auc) else "ROC-AUC        : unavailable")
        print(f"Training time  : {training_time:.2f} s")
        print(f"Inference time : {inference_time:.2f} s")
        print("Confusion matrix:\n", cm)

        results.append({
            "Experiment": experiment,
            "Model": "VQC",
            "Encoding": name,
            "Qubits": num_qubits,
            "Training samples": len(y_train),
            "Testing samples": len(y_test),
            "Features": len(FEATURES),
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "ROC_AUC": roc_auc,
            "Training time (s)": training_time,
            "Inference time (s)": inference_time,
            "Optimizer maxiter": MAX_ITER,
        })
        accuracy_by_encoding[name] = accuracy

        fig, ax = plt.subplots(figsize=(7, 6))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=encoder.classes_
        )
        disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
        ax.set_title(f"VQC {name}: {experiment.capitalize()}")
        fig.tight_layout()
        fig.savefig(
            os.path.join(
                OUTPUT_DIR,
                f"{experiment}_{name.lower().replace(' ', '_')}_confusion_matrix.png"
            ),
            dpi=250
        )
        plt.close(fig)

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(OUTPUT_DIR, f"{experiment}_vqc_results.csv"), index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(list(accuracy_by_encoding.keys()), list(accuracy_by_encoding.values()))
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.set_title(f"VQC Encoding Comparison: {experiment.capitalize()}")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f"{experiment}_vqc_encoding_comparison.png"), dpi=250)
    plt.close(fig)

    return results_df


if __name__ == "__main__":
    all_results = []
    for exp in EXPERIMENTS:
        all_results.append(evaluate_experiment(exp))

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(os.path.join(OUTPUT_DIR, "vqc_all_results.csv"), index=False)

    print("\nFINAL VQC COMPARISON")
    print(combined.to_string(index=False))
    print(f"\nAll results and plots saved in: {OUTPUT_DIR}")
