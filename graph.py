
import numpy as np
import matplotlib.pyplot as plt

# AUC values from your experiment outputs
results = {
    "Angle Encoding (QSVC)": 0.94,
    "Amplitude Encoding (QSVC)": 0.92,
    "ZZFeatureMap (QSVC)": 0.8167
}

# Generate illustrative ROC curves
# These match the approximate AUC values, not actual model predictions.

plt.figure(figsize=(9, 7))

fpr = np.linspace(0, 1, 1000)

for name, target_auc in results.items():
    alpha = (1 / target_auc) - 1
    tpr = fpr ** alpha

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"{name} (AUC = {target_auc:.4f})"
    )

plt.plot(
    [0, 1], [0, 1],
    linestyle="--",
    color="gray",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison of Three Quantum Encodings")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("roc_curve_comparison.png", dpi=300)
plt.show()
