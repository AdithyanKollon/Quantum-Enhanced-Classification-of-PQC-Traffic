
import os
import matplotlib
matplotlib.use("Agg")

import pandas as pd
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import StatePreparation, zz_feature_map

# Create output directory
OUTPUT_DIR = "quantum_circuits"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Four input features
x = ParameterVector("x", 4)

circuits = {}

# =====================================================
# 1. ANGLE ENCODING: 4 FEATURES, 4 QUBITS
# =====================================================
angle = QuantumCircuit(4, name="Angle Encoding")

for i in range(4):
    angle.ry(x[i], i)

circuits["Angle_Encoding"] = angle


# =====================================================
# 2. AMPLITUDE ENCODING: 4 FEATURES, 2 QUBITS
# =====================================================
amplitude = QuantumCircuit(2, name="Amplitude Encoding")

# Illustrative normalized vector.
# Replace this with an actual normalized feature vector
# if you want to visualize a specific dataset sample.
amplitude.append(
    StatePreparation([0.5, 0.5, 0.5, 0.5]),
    [0, 1]
)

circuits["Amplitude_Encoding"] = amplitude


# =====================================================
# 3. ZZ FEATURE MAP: 4 FEATURES, 4 QUBITS
# =====================================================
zz = zz_feature_map(
    feature_dimension=4,
    reps=2,
    entanglement="linear"
)

circuits["ZZFeatureMap"] = zz


# =====================================================
# DRAW, DECOMPOSE, TRANSPILE, AND MEASURE
# =====================================================
results = []

for name, circuit in circuits.items():

    print(f"\nProcessing {name}...")

    # Original high-level circuit
    circuit.draw(
        output="mpl",
        filename=f"{OUTPUT_DIR}/{name}_original.png",
        fold=-1
    )

    # Decompose composite operations
    decomposed = circuit.decompose(reps=5)

    decomposed.draw(
        output="mpl",
        filename=f"{OUTPUT_DIR}/{name}_decomposed.png",
        fold=-1
    )

    # Transpile with consistent settings
    compiled = transpile(
        circuit,
        basis_gates=["u", "cx"],
        optimization_level=0
    )

    compiled.draw(
        output="mpl",
        filename=f"{OUTPUT_DIR}/{name}_transpiled.png",
        fold=-1
    )

    # Display resource measurements
    print("Qubits:", circuit.num_qubits)
    print("Original depth:", circuit.depth())
    print("Original gate count:", circuit.size())
    print("Decomposed depth:", decomposed.depth())
    print("Decomposed gate count:", decomposed.size())
    print("Transpiled depth:", compiled.depth())
    print("Transpiled gate count:", compiled.size())
    print("Transpiled gate types:", dict(compiled.count_ops()))

    results.append({
        "Encoding": name,
        "Qubits": circuit.num_qubits,
        "Original Depth": circuit.depth(),
        "Original Gates": circuit.size(),
        "Decomposed Depth": decomposed.depth(),
        "Decomposed Gates": decomposed.size(),
        "Transpiled Depth": compiled.depth(),
        "Transpiled Gates": compiled.size(),
        "Transpiled Gate Types": str(dict(compiled.count_ops()))
    })

# Save measurements
df = pd.DataFrame(results)
df.to_csv(
    f"{OUTPUT_DIR}/circuit_resources.csv",
    index=False
)

print("\nAll circuit diagrams and measurements saved in:")
print(os.path.abspath(OUTPUT_DIR))