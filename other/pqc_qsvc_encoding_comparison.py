import sys
sys.stdout.reconfigure(encoding="utf-8")

import time
import numpy as np
import pandas as pd

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZZFeatureMap, StatePreparation
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

# ============================================================
# CONFIGURATION
# ============================================================
TRAIN_SAMPLES = 1000
TEST_SAMPLES = 500
RANDOM_STATE = 42

X_TRAIN_PATH = "PQC_X_train_scaled.csv"
X_TEST_PATH = "PQC_X_test_scaled.csv"
Y_TRAIN_PATH = "PQC_y_train.csv"
Y_TEST_PATH = "PQC_y_test.csv"

REMOVE_FEATURES = [
    "e2c_total_bytes",
    "e2_client_size",
    "e2_client_record_len",
    "e2_server_record_len",
]

# ============================================================
# TRUE AMPLITUDE-ENCODING FIDELITY KERNEL
# StatePreparation prepares each normalized input vector as amplitudes.
# Four features require 2 qubits (4 amplitudes).
# ============================================================
class AmplitudeFidelityKernel:
    def __init__(self, normalize=True):
        self.normalize = normalize

    def _state(self, row):
        row = np.asarray(row, dtype=float)
        # StatePreparation needs a vector of length 2**n and unit norm.
        norm = np.linalg.norm(row)
        if norm == 0:
            # Deterministic valid state for an all-zero feature vector.
            row = np.zeros_like(row)
            row[0] = 1.0
        else:
            row = row / norm

        n_qubits = int(np.ceil(np.log2(len(row))))
        dimension = 2**n_qubits
        if len(row) < dimension:
            row = np.pad(row, (0, dimension - len(row)))
            row = row / np.linalg.norm(row)

        circuit = QuantumCircuit(n_qubits)
        circuit.append(StatePreparation(row), range(n_qubits))
        return Statevector.from_instruction(circuit).data

    def evaluate(self, x_vec, y_vec=None):
        x_vec = np.asarray(x_vec, dtype=float)
        y_vec = x_vec if y_vec is None else np.asarray(y_vec, dtype=float)

        states_x = [self._state(row) for row in x_vec]
        states_y = states_x if y_vec is x_vec else [self._state(row) for row in y_vec]

        # Vectorized pairwise fidelities: much faster than nested Python loops.
        states_x = np.asarray(states_x)
        states_y = np.asarray(states_y)
        overlaps = states_x.conj() @ states_y.T
        return np.abs(overlaps) ** 2

    def evaluate_duplicates(self, x_vec, y_vec=None):
        # Compatibility helper used by some Qiskit ML kernel paths.
        return self.evaluate(x_vec, y_vec)


def make_angle_feature_map(num_features):
    x = ParameterVector("x", num_features)
    circuit = QuantumCircuit(num_features)
    for i in range(num_features):
        circuit.ry(x[i], i)
        circuit.rz(x[i], i)
    return circuit


def make_zz_feature_map(num_features):
    return ZZFeatureMap(
        feature_dimension=num_features,
        reps=2,
        entanglement="linear",
    )


# ============================================================
# LOAD AND PREPARE DATA
# ============================================================
print("Loading dataset...")

X_train_df = pd.read_csv(X_TRAIN_PATH)
X_test_df = pd.read_csv(X_TEST_PATH)
y_train_series = pd.read_csv(Y_TRAIN_PATH).squeeze()
y_test_series = pd.read_csv(Y_TEST_PATH).squeeze()

X_train_df = X_train_df.drop(
    columns=[c for c in REMOVE_FEATURES if c in X_train_df.columns]
)
X_test_df = X_test_df.drop(
    columns=[c for c in REMOVE_FEATURES if c in X_test_df.columns]
)

features = [c for c in X_train_df.columns if c in X_test_df.columns]
X_train_df = X_train_df[features]
X_test_df = X_test_df[features]

if len(X_train_df) < TRAIN_SAMPLES:
    raise ValueError(
        f"Training CSV has {len(X_train_df)} rows; need {TRAIN_SAMPLES}."
    )
if len(X_test_df) < TEST_SAMPLES:
    raise ValueError(
        f"Testing CSV has {len(X_test_df)} rows; need {TEST_SAMPLES}."
    )

X_train_df, _, y_train_series, _ = train_test_split(
    X_train_df, y_train_series,
    train_size=TRAIN_SAMPLES,
    stratify=y_train_series,
    random_state=RANDOM_STATE,
)
_, X_test_df, _, y_test_series = train_test_split(
    X_test_df, y_test_series,
    test_size=TEST_SAMPLES,
    stratify=y_test_series,
    random_state=RANDOM_STATE,
)

X_train = X_train_df.to_numpy(dtype=float)
X_test = X_test_df.to_numpy(dtype=float)
y_train = y_train_series.to_numpy(dtype=int)
y_test = y_test_series.to_numpy(dtype=int)

# Keep angle-based inputs in a sensible rotation range.
# The amplitude kernel independently normalizes each sample vector.
X_train_angles = np.clip(X_train, -np.pi, np.pi)
X_test_angles = np.clip(X_test, -np.pi, np.pi)

print("Features:", features)
print("Training samples:", len(y_train))
print("Testing samples:", len(y_test))
print("Number of input features:", X_train.shape[1])

# ============================================================
# EXPERIMENT RUNNER
# ============================================================
def run_qsvc(name, quantum_kernel, Xtr, Xte, ytr, yte):
    print("\n" + "=" * 72)
    print("Running:", name)
    print("=" * 72)

    try:
        model = QSVC(quantum_kernel=quantum_kernel)

        start = time.perf_counter()
        model.fit(Xtr, ytr)
        fit_seconds = time.perf_counter() - start

        start = time.perf_counter()
        predictions = model.predict(Xte)
        predict_seconds = time.perf_counter() - start

        start = time.perf_counter()
        scores = model.decision_function(Xte)
        score_seconds = time.perf_counter() - start

        result = {
            "encoding": name,
            "accuracy": accuracy_score(yte, predictions),
            "precision": precision_score(yte, predictions, zero_division=0),
            "recall": recall_score(yte, predictions, zero_division=0),
            "f1_score": f1_score(yte, predictions, zero_division=0),
            "roc_auc": roc_auc_score(yte, scores),
            "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
            "decision_score_seconds": score_seconds,
            "train_samples": len(ytr),
            "test_samples": len(yte),
            "features": Xtr.shape[1],
            "error": "",
        }

        print(f"Accuracy             : {result['accuracy'] * 100:.2f}%")
        print(f"Precision            : {result['precision'] * 100:.2f}%")
        print(f"Recall               : {result['recall'] * 100:.2f}%")
        print(f"F1-score             : {result['f1_score'] * 100:.2f}%")
        print(f"ROC-AUC              : {result['roc_auc'] * 100:.2f}%")
        print(f"Fit time             : {fit_seconds:.2f} s")
        print(f"Prediction time      : {predict_seconds:.2f} s")
        print(f"Decision-score time  : {score_seconds:.2f} s")
        print("Confusion matrix:")
        print(confusion_matrix(yte, predictions))
        return result

    except Exception as exc:
        print(f"FAILED: {name}: {type(exc).__name__}: {exc}")
        return {
            "encoding": name,
            "error": f"{type(exc).__name__}: {exc}",
            "train_samples": len(ytr),
            "test_samples": len(yte),
        }


# ============================================================
# RUN ENCODING COMPARISON
# ============================================================
results = []

# 1. Angle encoding: four features, four qubits.
results.append(
    run_qsvc(
        "Angle encoding",
        FidelityQuantumKernel(
            feature_map=make_angle_feature_map(X_train_angles.shape[1])
        ),
        X_train_angles, X_test_angles, y_train, y_test
    )
)

# 2. Amplitude encoding: four features prepared as amplitudes on two qubits.
# This uses a custom statevector fidelity kernel, not hardware execution.
results.append(
    run_qsvc(
        "Amplitude encoding",
        AmplitudeFidelityKernel(),
        X_train, X_test, y_train, y_test
    )
)

# 3. ZZFeatureMap: four features, four qubits.
results.append(
    run_qsvc(
        "ZZFeatureMap",
        FidelityQuantumKernel(
            feature_map=make_zz_feature_map(X_train_angles.shape[1])
        ),
        X_train_angles, X_test_angles, y_train, y_test
    )
)

# ============================================================
# SAVE RESULTS
# ============================================================
results_df = pd.DataFrame(results)
results_df.to_csv("qsvc_encoding_comparison.csv", index=False)

print("\n" + "=" * 72)
print("FINAL QSVC ENCODING COMPARISON")
print("=" * 72)
print(results_df.to_string(index=False))
print("\nSaved results to qsvc_encoding_comparison.csv")
