
import sys
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import numpy as np
import time

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# ============================================================
# 1. LOAD DATA
# ============================================================

X_train = pd.read_csv("PQC_X_train_scaled.csv")
X_test = pd.read_csv("PQC_X_test_scaled.csv")

y_train = pd.read_csv("PQC_y_train.csv").squeeze()
y_test = pd.read_csv("PQC_y_test.csv").squeeze()

# ============================================================
# 2. FEATURE ABLATION
# ============================================================

remove_features = [
    "e2c_total_bytes",
    "e2_client_size",
    "e2_client_record_len",
    "e2_server_record_len"
]

X_train = X_train.drop(columns=remove_features)
X_test = X_test.drop(columns=remove_features)

assert X_train.shape[1] == 4, "Expected exactly four features"
assert list(X_train.columns) == list(X_test.columns)

print("Features:", list(X_train.columns))

# ============================================================
# 3. SELECT 100 TRAIN + 100 TEST SAMPLES
# ============================================================

X_train, _, y_train, _ = train_test_split(
    X_train,
    y_train,
    train_size=100,
    stratify=y_train,
    random_state=42
)

_, X_test, _, y_test = train_test_split(
    X_test,
    y_test,
    test_size=100,
    stratify=y_test,
    random_state=42
)

X_train = X_train.to_numpy(dtype=float)
X_test = X_test.to_numpy(dtype=float)

y_train = y_train.to_numpy(dtype=int)
y_test = y_test.to_numpy(dtype=int)

print("\nTraining samples:", len(y_train))
print("Testing samples:", len(y_test))
print("Total samples:", len(y_train) + len(y_test))

# ============================================================
# 4. DEFINE THREE ENCODING METHODS
# ============================================================

# A. ANGLE ENCODING
# Four features -> four qubits

def create_angle_feature_map():
    x = ParameterVector("x", 4)
    circuit = QuantumCircuit(4)

    for i in range(4):
        circuit.ry(x[i], i)

    for i in range(3):
        circuit.cz(i, i + 1)

    return circuit


angle_map = create_angle_feature_map()

# B. AMPLITUDE ENCODING
# Four features -> two qubits
# Kernel = squared overlap of normalized amplitude states.
# This computes the exact amplitude-encoding fidelity kernel.

def amplitude_kernel(A, B):
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)

    norm_A = np.linalg.norm(A, axis=1, keepdims=True)
    norm_B = np.linalg.norm(B, axis=1, keepdims=True)

    if np.any(norm_A == 0) or np.any(norm_B == 0):
        raise ValueError(
            "Amplitude encoding cannot normalize zero vectors."
        )

    A_norm = A / norm_A
    B_norm = B / norm_B

    overlaps = A_norm @ B_norm.T

    return np.square(overlaps)


# C. ZZ FEATURE MAP
# Four features -> four qubits

zz_map = ZZFeatureMap(
    feature_dimension=4,
    reps=2,
    entanglement="linear"
)

# ============================================================
# 5. CREATE QUANTUM KERNELS
# ============================================================

angle_kernel = FidelityQuantumKernel(
    feature_map=angle_map
)

zz_kernel = FidelityQuantumKernel(
    feature_map=zz_map
)

# ============================================================
# 6. TRAIN AND EVALUATE EACH ENCODING
# ============================================================

results = []

encodings = [
    ("Angle Encoding", angle_kernel),
    ("Amplitude Encoding", None),
    ("ZZFeatureMap", zz_kernel)
]

for name, quantum_kernel in encodings:

    print("\n" + "=" * 65)
    print("ENCODING:", name)
    print("=" * 65)

    start_time = time.time()

    # Compute training and testing kernel matrices
    if name == "Amplitude Encoding":
        K_train = amplitude_kernel(X_train, X_train)
        K_test = amplitude_kernel(X_test, X_train)
    else:
        K_train = quantum_kernel.evaluate(
            x_vec=X_train
        )
        K_test = quantum_kernel.evaluate(
            x_vec=X_test,
            y_vec=X_train
        )

    kernel_time = time.time() - start_time

    # Train SVM on the kernel matrices
    model = SVC(
        kernel="precomputed",
        probability=False
    )

    train_start = time.time()
    model.fit(K_train, y_train)
    training_time = time.time() - train_start

    # Predict
    prediction_start = time.time()

    y_pred = model.predict(K_test)
    y_score = model.decision_function(K_test)

    prediction_time = time.time() - prediction_start

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, zero_division=0
    )
    recall = recall_score(
        y_test, y_pred, zero_division=0
    )
    f1 = f1_score(
        y_test, y_pred, zero_division=0
    )
    roc_auc = roc_auc_score(y_test, y_score)

    cm = confusion_matrix(
        y_test, y_pred, labels=[0, 1]
    )

    results.append({
        "Encoding": name,
        "Accuracy (%)": round(accuracy * 100, 2),
        "Precision (%)": round(precision * 100, 2),
        "Recall (%)": round(recall * 100, 2),
        "F1-score (%)": round(f1 * 100, 2),
        "ROC-AUC (%)": round(roc_auc * 100, 2),
        "Kernel time (s)": round(kernel_time, 2),
        "Training time (s)": round(training_time, 2),
        "Inference time (s)": round(prediction_time, 2)
    })

    print(f"Accuracy       : {accuracy * 100:.2f}%")
    print(f"Precision      : {precision * 100:.2f}%")
    print(f"Recall         : {recall * 100:.2f}%")
    print(f"F1-score       : {f1 * 100:.2f}%")
    print(f"ROC-AUC        : {roc_auc * 100:.2f}%")
    print(f"Kernel time    : {kernel_time:.2f} seconds")
    print(f"Training time  : {training_time:.2f} seconds")
    print(f"Inference time : {prediction_time:.2f} seconds")

    print("\nConfusion Matrix:")
    print(cm)
    print("Class mapping: 0 = Non-PQC, 1 = PQC")

# ============================================================
# 7. COMPARE ALL THREE ENCODINGS
# ============================================================

results_df = pd.DataFrame(results)

print("\n\n" + "=" * 90)
print("FINAL COMPARISON: THREE QUANTUM ENCODINGS")
print("=" * 90)

print(results_df.to_string(index=False))

results_df.to_csv(
    "QSVM_3_encoding_comparison_100_samples.csv",
    index=False
)

print("\nResults saved to:")
print("QSVM_3_encoding_comparison_100_samples.csv")
