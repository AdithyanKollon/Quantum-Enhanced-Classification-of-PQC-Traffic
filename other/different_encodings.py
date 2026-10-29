
import time
import numpy as np
import pandas as pd

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import ZZFeatureMap
from qiskit.circuit.library import StatePreparation

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# ==================================================
# 1. LOAD DATA
# ==================================================
# Load ALL training and testing samples
# X_train_full = pd.read_csv("PQC_X_train.csv")
# y_train_full = pd.read_csv("PQC_y_train.csv").squeeze()

# X_test_full = pd.read_csv("PQC_X_test.csv")
# y_test_full = pd.read_csv("PQC_y_test.csv").squeeze()

# # Select the same four features
# features = [
#     "e2_client_record_len",
#     "e2c_total_bytes",
#     "e2_client_size",
#     "e2_server_record_len"
# ]

# X_train = X_train_full[features].reset_index(drop=True)
# y_train = y_train_full.reset_index(drop=True)

# X_test = X_test_full[features].reset_index(drop=True)
# y_test = y_test_full.reset_index(drop=True)

# # Convert labels to NumPy arrays
# y_train = np.asarray(y_train)
# y_test = np.asarray(y_test)

# print("Training samples:", len(X_train))  # 30000
# print("Testing samples:", len(X_test))    # 7500
# Load existing train/test splits
X_train_full = pd.read_csv("PQC_X_train.csv")
y_train_full = pd.read_csv("PQC_y_train.csv").squeeze()

X_test_full = pd.read_csv("PQC_X_test.csv")
y_test_full = pd.read_csv("PQC_y_test.csv").squeeze()

# Select the same four features
features = [
    "e2_client_record_len",
    "e2c_total_bytes",
    "e2_client_size",
    "e2_server_record_len"
]

X_train_full = X_train_full[features].reset_index(drop=True)
y_train_full = pd.Series(y_train_full).reset_index(drop=True)

X_test_full = X_test_full[features].reset_index(drop=True)
y_test_full = pd.Series(y_test_full).reset_index(drop=True)

# Select 2000 training samples
X_train, _, y_train, _ = train_test_split(
    X_train_full,
    y_train_full,
    train_size=2000,
    random_state=42,
    stratify=y_train_full
)

# Select 1000 testing samples from the held-out test set
X_test, _, y_test, _ = train_test_split(
    X_test_full,
    y_test_full,
    train_size=1000,
    random_state=42,
    stratify=y_test_full
)

y_train = np.asarray(y_train)
y_test = np.asarray(y_test)

print("Training samples:", len(X_train))  # 2000
print("Testing samples:", len(X_test))    # 1000
# ==================================================
# 2. PREPROCESSING
# Fit scalers only on the selected training samples
# ==================================================
angle_scaler = MinMaxScaler(feature_range=(0, np.pi))
X_train_angle = angle_scaler.fit_transform(X_train)
X_test_angle = angle_scaler.transform(X_test)

amp_scaler = MinMaxScaler(feature_range=(0, 1))
X_train_amp = amp_scaler.fit_transform(X_train)
X_test_amp = amp_scaler.transform(X_test)

# ZZFeatureMap also receives bounded numerical values
zz_scaler = MinMaxScaler(feature_range=(0, np.pi))
X_train_zz = zz_scaler.fit_transform(X_train)
X_test_zz = zz_scaler.transform(X_test)


# ==================================================
# 3. ANGLE ENCODING
# Four features -> four qubits
# ==================================================
def angle_state(sample):
    qc = QuantumCircuit(4)

    for i, value in enumerate(sample):
        qc.ry(float(value), i)

    return Statevector.from_instruction(qc).data


# ==================================================
# 4. AMPLITUDE ENCODING
# Four features -> two qubits
# ==================================================
def amplitude_state(sample):
    norm = np.linalg.norm(sample)

    if norm == 0:
        normalized = np.array([1.0, 0.0, 0.0, 0.0])
    else:
        normalized = sample / norm

    qc = QuantumCircuit(2)
    qc.append(StatePreparation(normalized), [0, 1])

    return Statevector.from_instruction(qc).data


# ==================================================
# 5. ZZ FEATURE MAP
# Four features -> four qubits
# ==================================================
zz_map = ZZFeatureMap(
    feature_dimension=4,
    reps=2,
    entanglement="linear"
)

def zz_state(sample):
    bound_circuit = zz_map.assign_parameters(sample)
    return Statevector.from_instruction(bound_circuit).data


# ==================================================
# 6. FIDELITY KERNEL
# ==================================================
def fidelity_kernel(X1, X2, encoder):
    states1 = [encoder(row) for row in X1]
    states2 = [encoder(row) for row in X2]

    K = np.zeros((len(states1), len(states2)))

    for i, state1 in enumerate(states1):
        for j, state2 in enumerate(states2):
            K[i, j] = abs(np.vdot(state1, state2)) ** 2

    return K


# ==================================================
# 7. TRAIN AND EVALUATE QSVM
# ==================================================
def evaluate_encoding(name, encoder, Xtr, Xte, qubits):
    print(f"\nRunning {name}...")

    start = time.perf_counter()
    K_train = fidelity_kernel(Xtr, Xtr, encoder)
    K_test = fidelity_kernel(Xte, Xtr, encoder)
    kernel_time = time.perf_counter() - start

    model = SVC(
        kernel="precomputed",
        class_weight="balanced"
    )

    start = time.perf_counter()
    model.fit(K_train, y_train)
    training_time = time.perf_counter() - start

    predictions = model.predict(K_test)
    decision_scores = model.decision_function(K_test)

    return {
        "Encoding": name,
        "Qubits": qubits,
        "Train_Samples": len(y_train),
        "Test_Samples": len(y_test),
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1": f1_score(
            y_test, predictions, zero_division=0
        ),
        "ROC_AUC": roc_auc_score(y_test, decision_scores),
        "Kernel_Time_s": kernel_time,
        "SVM_Training_Time_s": training_time
    }


# ==================================================
# 8. RUN ALL THREE METHODS
# ==================================================
results = []

results.append(evaluate_encoding(
    "Angle Encoding",
    angle_state,
    X_train_angle,
    X_test_angle,
    4
))

results.append(evaluate_encoding(
    "Amplitude Encoding",
    amplitude_state,
    X_train_amp,
    X_test_amp,
    2
))

results.append(evaluate_encoding(
    "ZZFeatureMap",
    zz_state,
    X_train_zz,
    X_test_zz,
    4
))

results_df = pd.DataFrame(results)

print("\n========== FINAL ENCODING COMPARISON ==========")
print(results_df.round(4).to_string(index=False))


