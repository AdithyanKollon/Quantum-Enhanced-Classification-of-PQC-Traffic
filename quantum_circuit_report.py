"""
Draw the exact feature-map circuits used for the QSVM-style comparison and
print reproducible circuit resource counts.

Install:
    python -m pip install qiskit qiskit-machine-learning matplotlib pylatexenc numpy

Run:
    python quantum_circuit_report.py

Outputs:
    quantum_circuits/angle_encoding.png
    quantum_circuits/amplitude_encoding.png
    quantum_circuits/zz_feature_map.png
    quantum_circuits/circuit_resources.txt

Important:
- Angle encoding and ZZFeatureMap follow the configurations described in the
  current experiment: 4 features, 4 qubits, ZZFeatureMap reps=2.
- Amplitude encoding below uses Qiskit's StatePreparation with a normalized
  four-element illustrative vector, which encodes 4 amplitudes on 2 qubits.
- Replace the illustrative values with a real preprocessed row if you want
  to show one example input. Resource counts can depend on Qiskit version,
  transpiler settings, basis gates, and backend.
- This script draws feature maps. It does not draw a complete QSVM training
  workflow or a VQC trainable ansatz.
"""

from pathlib import Path
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import ZZFeatureMap, StatePreparation

OUT = Path("quantum_circuits")
OUT.mkdir(parents=True, exist_ok=True)

# Example scaled feature values for illustration only.
# Four values correspond to the four selected QML features:
# e4_entropy_h, e5_entropy_c, e6_time_char, e6b_flow_duration_ms.
x = [0.35, 1.10, 2.00, 2.65]


def save_circuit_image(circuit, filename, title):
    """Save circuit as a PNG using Qiskit's Matplotlib drawer."""
    fig = circuit.draw(output="mpl", fold=-1)
    fig.suptitle(title, fontsize=12)
    fig.savefig(OUT / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved diagram: {OUT / filename}")


def report_resources(handle, circuit, label, f):
    """Write circuit depth and operation counts to a text file."""
    f.write(f"\n{label}\n")
    f.write("-" * len(label) + "\n")
    f.write(f"Qubits: {circuit.num_qubits}\n")
    f.write(f"Original circuit depth: {circuit.depth()}\n")
    f.write(f"Original operation counts: {dict(circuit.count_ops())}\n")

    # Generic basis-gate transpilation is for reproducible inspection only.
    # This is not necessarily the same backend/basis used in the experiments.
    try:
        tc = transpile(circuit, optimization_level=1)
        f.write("Transpiled using local Qiskit default target (inspection only)\n")
        f.write(f"Transpiled depth: {tc.depth()}\n")
        f.write(f"Transpiled operation counts: {dict(tc.count_ops())}\n")
    except Exception as exc:
        f.write(f"Transpilation inspection failed: {exc}\n")

    f.write(f"OpenQASM export is not required for this report.\n")


def main():
    circuits = {}

    # 1) Angle encoding: each feature controls an Ry rotation on one qubit.
    angle = QuantumCircuit(4, name="AngleEncoding")
    for i, value in enumerate(x):
        angle.ry(float(value), i)
    circuits["Angle encoding"] = angle
    save_circuit_image(
        angle,
        "angle_encoding.png",
        "Angle encoding: 4 features, 4 qubits"
    )

    # 2) Amplitude encoding: normalize four values and prepare a 2-qubit state.
    amplitudes = np.asarray([0.1, 0.3, 0.5, 0.7], dtype=float)
    amplitudes = amplitudes / np.linalg.norm(amplitudes)
    amplitude = QuantumCircuit(2, name="AmplitudeEncoding")
    amplitude.append(StatePreparation(amplitudes), [0, 1])
    circuits["Amplitude encoding"] = amplitude
    save_circuit_image(
        amplitude,
        "amplitude_encoding.png",
        "Amplitude encoding: 4 normalized amplitudes, 2 qubits"
    )

    # 3) ZZFeatureMap: four features, four qubits, two repetitions.
    zz = ZZFeatureMap(
        feature_dimension=4,
        reps=2,
        entanglement="full"
    )
    circuits["ZZFeatureMap"] = zz
    save_circuit_image(
        zz,
        "zz_feature_map.png",
        "ZZFeatureMap: 4 features, 4 qubits, reps=2"
    )

    # Print and save resources for the original and inspected transpiled circuits.
    report_path = OUT / "circuit_resources.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("QUANTUM CIRCUIT RESOURCE REPORT\n")
        f.write("These are measurements from this script's circuits and Qiskit environment.\n")
        f.write("Transpilation uses the local Qiskit default target, not necessarily the experiment backend.\n")
        for label, circuit in circuits.items():
            report_resources(None, circuit, label, f)

    print(f"\nSaved resource report: {report_path}")
    print("\nCircuit resource summary (original circuits):")
    for label, circuit in circuits.items():
        print(f"{label}: qubits={circuit.num_qubits}, "
              f"depth={circuit.depth()}, ops={dict(circuit.count_ops())}")

    print("\nNOTE: use these resource values in your report only if these circuits")
    print("match your experiment implementation and the stated transpilation settings.")


if __name__ == "__main__":
    main()
