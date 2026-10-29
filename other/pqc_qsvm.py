
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZZFeatureMap, StatePreparation
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
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

print("Features used:", list(X_train.columns))

# ============================================================
# 3. SELECT IDENTICAL STRATIFIED SUBSETS
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
    test_size=50,
    stratify=y_test,
    random_state=42
)

y_train = np.asarray(y_train, dtype=int)
y_test = np.asarray(y_test, dtype=int)

# Fit separate preprocessing on training data only.
# Angle and ZZ use values in [0, pi].
angle_scaler = MinMaxScaler(feature_range=(0, np.pi))
X_train_angle = angle_scaler.fit_transform(X_train)
X_test_angle = angle_scaler.transform(X_test)

# Amplitude encoding uses non-negative vectors that are normalized
# separately to unit length before state preparation.
amp_scaler = MinMaxScaler(feature_range=(0, 1))
X_train_amp = amp_scaler.fit_transform(X_train)
X_test_amp = amp_scaler.transform(X_test)

def normalize_rows(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    # Handle a possible all-zero row safely.
    zero_rows = norms[:, 0] == 0
    norms[zero_rows] = 1.0
    result = X / norms
    result[zero_rows, 0] = 1.0
    return result

X_train_amp = normalize_rows(X_train_amp)
X_test_amp = normalize_rows(X_test_amp)

print("Training samples:", len(y_train))
print("Testing samples:", len(y_test))
print("Features:", X_train.shape[1])
print("Class mapping: 0 = Non-PQC, 1 = PQC")

# ============================================================
# 4. DEFINE FEATURE MAPS
# ============================================================

n_features = X_train.shape[1]

# Angle encoding: one RY rotation per feature.
def create_angle_feature_map(n):
    params = ParameterVector("x", n)
    circuit = QuantumCircuit(n)

    for i in range(n):
        circuit.ry(params[i], i)

    return circuit

angle_map = create_angle_feature_map(n_features)

# ZZFeatureMap: four qubits, two repetitions, linear entanglement.
zz_map = ZZFeatureMap(
    feature_dimension=n_features,
    reps=2,
    entanglement="linear"
)

# ============================================================
# 5. MEASURE CIRCUIT RESOURCE INFORMATION
# ============================================================

def circuit_info(name, circuit):
    decomposed = circuit.decompose()
    transpiled_circuit = transpile(
        circuit,
        basis_gates=["rz", "sx", "x", "cx"],
        optimization_level=0
    )

    return {
        "Encoding": name,
        "Qubits": circuit.num_qubits,
        "Original Depth": circuit.depth(),
        "Decomposed Depth": decomposed.depth(),
        "Transpiled Depth": transpiled_circuit.depth(),
        "Gate Counts": str(dict(transpiled_circuit.count_ops()))
    }

resource_results = [
    circuit_info("Angle encoding", angle_map),
    circuit_info("ZZFeatureMap", zz_map)
]

# ============================================================
# 6. HELPER: EVALUATE PARAMETERIZED FIDELITY-KERNEL QSVC
# ============================================================

results = []
roc_data = {}

def run_qsvc(name, feature_map, Xtr, Xte, qubits):
    print("\n" + "=" * 70)
    print(f"QSVC — {name}")
    print("=" * 70)

    kernel = FidelityQuantumKernel(feature_map=feature_map)
    model = QSVC(quantum_kernel=kernel)

    start = time.perf_counter()
    model.fit(Xtr, y_train)
    training_time = time.perf_counter() - start

    start = time.perf_counter()
    y_pred = model.predict(Xte)
    y_score = model.decision_function(Xte)
    inference_time = time.perf_counter() - start

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc_value = roc_auc_score(y_test, y_score)
    cm = confusion_matrix(y_test, y_pred)

    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_data[name] = (fpr, tpr, auc_value)

    print(f"Accuracy       : {accuracy * 100:.2f}%")
    print(f"Precision      : {precision * 100:.2f}%")
    print(f"Recall         : {recall * 100:.2f}%")
    print(f"F1-score       : {f1 * 100:.2f}%")
    print(f"ROC-AUC        : {auc_value * 100:.2f}%")
    print(f"Training time  : {training_time:.4f} seconds")
    print(f"Inference time : {inference_time:.4f} seconds")
    print("Confusion matrix:\n", cm)

    results.append({
        "Encoding": name,
        "Qubits": qubits,
        "Accuracy (%)": accuracy * 100,
        "Precision (%)": precision * 100,
        "Recall (%)": recall * 100,
        "F1-score (%)": f1 * 100,
        "ROC-AUC (%)": auc_value * 100,
        "Training Time (s)": training_time,
        "Inference Time (s)": inference_time
    })

# ============================================================
# 7. RUN ANGLE ENCODING
# ============================================================

run_qsvc(
    "Angle encoding",
    angle_map,
    X_train_angle,
    X_test_angle,
    n_features
)

# ============================================================
# 8. RUN ZZFEATUREMAP
# ============================================================

run_qsvc(
    "ZZFeatureMap",
    zz_map,
    X_train_angle,
    X_test_angle,
    n_features
)

# ============================================================
# 9. AMPLITUDE ENCODING: FIDELITY KERNEL
# ============================================================
# Each four-feature normalized vector is prepared as a two-qubit
# quantum state. Statevector overlaps give exact fidelity kernels.
# SVC uses the precomputed quantum kernel.

def amplitude_state(vector):
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)

    if norm == 0:
        vector = np.zeros_like(vector)
        vector[0] = 1.0
    else:
        vector = vector / norm

    qc = QuantumCircuit(2)
    qc.append(StatePreparation(vector), [0, 1])
    return Statevector.from_instruction(qc).data

def amplitude_kernel_matrix(X1, X2):
    states1 = [amplitude_state(row) for row in X1]
    states2 = [amplitude_state(row) for row in X2]

    K = np.zeros((len(states1), len(states2)))

    for i, state1 in enumerate(states1):
        for j, state2 in enumerate(states2):
            K[i, j] = abs(np.vdot(state1, state2)) ** 2

    return K

print("\n" + "=" * 70)
print("QUANTUM KERNEL SVM — AMPLITUDE ENCODING")
print("=" * 70)

# Record representative state-preparation circuit resources.
example_vector = X_train_amp[0]
amp_qc = QuantumCircuit(2)
amp_qc.append(StatePreparation(example_vector), [0, 1])

amp_decomposed = amp_qc.decompose()
amp_transpiled = transpile(
    amp_qc,
    basis_gates=["rz", "sx", "x", "cx"],
    optimization_level=0
)

resource_results.append({
    "Encoding": "Amplitude encoding",
    "Qubits": 2,
    "Original Depth": amp_qc.depth(),
    "Decomposed Depth": amp_decomposed.depth(),
    "Transpiled Depth": amp_transpiled.depth(),
    "Gate Counts": str(dict(amp_transpiled.count_ops()))
})

print("Qubits:", amp_qc.num_qubits)
print("Representative state-preparation depth:", amp_qc.depth())
print("Decomposed depth:", amp_decomposed.depth())
print("Transpiled depth:", amp_transpiled.depth())
print("Gate counts:", amp_transpiled.count_ops())

# Measure kernel construction separately.
start = time.perf_counter()
K_train_amp = amplitude_kernel_matrix(X_train_amp, X_train_amp)
kernel_train_time = time.perf_counter() - start

start = time.perf_counter()
K_test_amp = amplitude_kernel_matrix(X_test_amp, X_train_amp)
kernel_test_time = time.perf_counter() - start

model_amp = SVC(kernel="precomputed")

start = time.perf_counter()
model_amp.fit(K_train_amp, y_train)
svm_fit_time = time.perf_counter() - start

start = time.perf_counter()
y_pred_amp = model_amp.predict(K_test_amp)
y_score_amp = model_amp.decision_function(K_test_amp)
prediction_time_amp = time.perf_counter() - start

accuracy = accuracy_score(y_test, y_pred_amp)
precision = precision_score(y_test, y_pred_amp, zero_division=0)
recall = recall_score(y_test, y_pred_amp, zero_division=0)
f1 = f1_score(y_test, y_pred_amp, zero_division=0)
auc_value = roc_auc_score(y_test, y_score_amp)
cm = confusion_matrix(y_test, y_pred_amp)

fpr, tpr, _ = roc_curve(y_test, y_score_amp)
roc_data["Amplitude encoding"] = (fpr, tpr, auc_value)

print(f"Accuracy              : {accuracy * 100:.2f}%")
print(f"Precision             : {precision * 100:.2f}%")
print(f"Recall                : {recall * 100:.2f}%")
print(f"F1-score              : {f1 * 100:.2f}%")
print(f"ROC-AUC               : {auc_value * 100:.2f}%")
print(f"Training kernel time  : {kernel_train_time:.4f} seconds")
print(f"Test kernel time      : {kernel_test_time:.4f} seconds")
print(f"SVM fit time          : {svm_fit_time:.4f} seconds")
print(f"Prediction time       : {prediction_time_amp:.4f} seconds")
print("Confusion matrix:\n", cm)

results.append({
    "Encoding": "Amplitude encoding",
    "Qubits": 2,
    "Accuracy (%)": accuracy * 100,
    "Precision (%)": precision * 100,
    "Recall (%)": recall * 100,
    "F1-score (%)": f1 * 100,
    "ROC-AUC (%)": auc_value * 100,
    "Training Time (s)": kernel_train_time + svm_fit_time,
    "Inference Time (s)": kernel_test_time + prediction_time_amp
})

# ============================================================
# 10. COMPARISON TABLE
# ============================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 110)
print("FINAL QUANTUM ENCODING COMPARISON — PQC vs NON-PQC")
print("=" * 110)
print(results_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

results_df.to_csv("PQC_QSVM_encoding_comparison.csv", index=False)

resources_df = pd.DataFrame(resource_results)
print("\nCIRCUIT RESOURCE COMPARISON")
print(resources_df.to_string(index=False))
resources_df.to_csv("PQC_QSVM_circuit_resources.csv", index=False)

# ============================================================
# 11. COMBINED ROC CURVE
# ============================================================

plt.figure(figsize=(8, 6))

for name, (fpr, tpr, auc_value) in roc_data.items():
    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"{name} (AUC = {auc_value:.3f})"
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random classifier (AUC = 0.500)"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("QSVM ROC Curves — PQC vs Non-PQC")
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("PQC_QSVM_encoding_ROC.png", dpi=300, bbox_inches="tight")
plt.show()

print("\nSaved:")
print("1. PQC_QSVM_encoding_comparison.csv")
print("2. PQC_QSVM_circuit_resources.csv")
print("3. PQC_QSVM_encoding_ROC.png")
