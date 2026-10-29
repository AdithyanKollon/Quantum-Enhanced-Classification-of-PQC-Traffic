"""
Draw representative quantum feature-encoding circuits for the PQC traffic project.

Install: pip install qiskit matplotlib
Run: python draw_quantum_circuits.py

Outputs are illustrative. Confirm circuit details against your final experiment scripts.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, StatePreparation

OUT = Path("quantum_circuits")
OUT.mkdir(exist_ok=True)

def save_circuit(qc, filename, title):
    fig = qc.draw(output="mpl", fold=-1)
    fig.suptitle(title, fontsize=13)
    fig.savefig(OUT / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT / filename}")

# Illustrative scaled values for four features
x = [0.35, 1.10, 2.00, 2.65]

# 1. Angle encoding: one feature mapped to one qubit rotation
angle = QuantumCircuit(4, name="Angle encoding")
for i, value in enumerate(x):
    angle.ry(float(value), i)
save_circuit(angle, "angle_encoding.png", "Angle encoding: 4 features, 4 qubits")

# 2. Amplitude encoding: four normalized amplitudes stored in a 2-qubit state
amplitudes = np.array([0.1, 0.3, 0.5, 0.7], dtype=float)
amplitudes = amplitudes / np.linalg.norm(amplitudes)
amp = QuantumCircuit(2, name="Amplitude encoding")
amp.append(StatePreparation(amplitudes), [0, 1])
save_circuit(amp, "amplitude_encoding.png", "Amplitude encoding: 4 amplitudes, 2 qubits")

# 3. ZZFeatureMap: feature rotations and pairwise interactions
zz = ZZFeatureMap(feature_dimension=4, reps=2, entanglement="full")
save_circuit(zz, "zz_feature_map.png", "ZZFeatureMap: 4 features, 4 qubits, reps=2")

print("\nCircuit diagrams generated.")
print("For exact report diagrams, confirm these against the final QSVM/VQC scripts.")
