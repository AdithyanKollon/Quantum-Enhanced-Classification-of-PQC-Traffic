
import sys
sys.stdout.reconfigure(encoding="utf-8")

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.algorithms import VQC
from qiskit_algorithms.optimizers import COBYLA

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
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

print("Features:", list(X_train.columns))

if X_train.shape[1] != 4:
    raise ValueError("Expected exactly 4 features after ablation.")

# ============================================================
# 3. SELECT 100 TRAINING + 100 TESTING SAMPLES
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
print("Class mapping: 0 = Non-PQC, 1 = PQC")

# ============================================================
# 4. AMPLITUDE ENCODING PREPROCESSING
# ============================================================

def normalize_amplitudes(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)

    if np.any(norms == 0):
        raise ValueError("Zero vector found in amplitude encoding.")

    return X / norms


X_train_amp = normalize_amplitudes(X_train)
X_test_amp = normalize_amplitudes(X_test)

# ============================================================
# 5. ANGLE ENCODING
# ============================================================

def create_angle_feature_map(n):
    x = ParameterVector("x", n)
    qc = QuantumCircuit(n)

    for i in range(n):
        qc.ry(x[i], i)

    for i in range(n - 1):
        qc.cx(i, i + 1)

    return qc


# ============================================================
# 6. AMPLITUDE ENCODING
# Four amplitudes encoded into two qubits
# ============================================================

def create_amplitude_feature_map():
    x = ParameterVector("x", 4)
    qc = QuantumCircuit(2)

    # Prepare a normalized four-dimensional quantum state.
    # State preparation is performed using an initialization
    # circuit with data-dependent amplitudes.
    qc.initialize([0.5, 0.5, 0.5, 0.5], [0, 1])

    # Data-dependent rotations provide a trainable encoding.
    # This is an amplitude-inspired encoding, not exact
    # arbitrary amplitude encoding of each input vector.
    qc.ry(x[0], 0)
    qc.ry(x[1], 1)
    qc.cx(0, 1)
    qc.ry(x[2], 0)
    qc.ry(x[3], 1)

    return qc


# ============================================================
# 7. DEFINE FEATURE MAPS
# ============================================================

feature_maps = {
    "Angle Encoding": create_angle_feature_map(4),
    "Amplitude Encoding": create_amplitude_feature_map(),
    "ZZFeatureMap": ZZFeatureMap(
        feature_dimension=4,
        reps=2,
        entanglement="linear"
    )
}

# ============================================================
# 8. TRAIN AND EVALUATE EACH VQC
# ============================================================

results = []
roc_data = {}

for name, feature_map in feature_maps.items():

    print("\n" + "=" * 65)
    print("VQC:", name)
    print("=" * 65)

    if name == "Amplitude Encoding":
        Xtr = X_train_amp
        Xte = X_test_amp
    else:
        Xtr = X_train
        Xte = X_test

    num_qubits = feature_map.num_qubits

    ansatz = RealAmplitudes(
        num_qubits=num_qubits,
        reps=1
    )

    optimizer = COBYLA(maxiter=30)

    model = VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=optimizer
    )

    # Train
    start = time.time()
    model.fit(Xtr, y_train)
    training_time = time.time() - start

    # Predict
    start = time.time()
    y_pred = np.asarray(model.predict(Xte)).reshape(-1)
    prediction_time = time.time() - start

    y_pred = y_pred.astype(int)

    # Obtain predicted probabilities for ROC
    probabilities = np.asarray(model.predict_proba(Xte))

    if probabilities.ndim == 2 and probabilities.shape[1] == 2:
        y_score = probabilities[:, 1]
    else:
        raise ValueError(
            f"Unexpected probability shape for {name}: "
            f"{probabilities.shape}"
        )

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

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_data[name] = (fpr, tpr, roc_auc)

    print(f"Qubits         : {num_qubits}")
    print(f"Accuracy       : {accuracy * 100:.2f}%")
    print(f"Precision      : {precision * 100:.2f}%")
    print(f"Recall         : {recall * 100:.2f}%")
    print(f"F1-score       : {f1 * 100:.2f}%")
    print(f"ROC-AUC        : {roc_auc:.4f}")
    print(f"Training time  : {training_time:.2f} seconds")
    print(f"Inference time : {prediction_time:.2f} seconds")
    print("Confusion matrix:")
    print(cm)

    results.append({
        "Encoding": name,
        "Qubits": num_qubits,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "ROC-AUC": roc_auc,
        "Training time": training_time,
        "Inference time": prediction_time
    })

# ============================================================
# 9. SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    "vqc_3_encoding_results.csv",
    index=False
)

print("\nFINAL VQC COMPARISON")
print(results_df.to_string(index=False))

# ============================================================
# 10. PLOT ROC CURVES
# ============================================================

plt.figure(figsize=(9, 7))

for name, (fpr, tpr, score) in roc_data.items():
    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"{name} (AUC = {score:.4f})"
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    color="gray",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("VQC ROC Curve Comparison: Three Encodings")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.tight_layout()

plt.savefig(
    "vqc_3_encoding_roc.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nSaved files:")
print("vqc_3_encoding_results.csv")
print("vqc_3_encoding_roc.png")
