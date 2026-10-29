import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42
N_BINARY = 200
N_MULTICLASS = 200
N_CLASSES = 6

OUTPUT_DIR = "experiment_data/roc_auc_curves"
os.makedirs(OUTPUT_DIR, exist_ok=True)

rng = np.random.default_rng(SEED)

# ============================================================
# ILLUSTRATIVE AUC TARGETS
# These are synthetic targets, NOT measured model results.
# Replace with actual prediction scores for real ROC curves.
# ============================================================

AUC_TARGETS = {
    "Classical ML": {
        "binary": 0.85,
        "multiclass": 0.75
    },
    "MLP": {
        "binary": 0.84,
        "multiclass": 0.70
    },
    "VQC": {
        "binary": {
            "Angle Encoding": 0.80,
            "Amplitude Encoding": 0.78,
            "ZZFeatureMap": 0.82
        },
        "multiclass": {
            "Angle Encoding": 0.72,
            "Amplitude Encoding": 0.68,
            "ZZFeatureMap": 0.74
        }
    },
    "QSVM": {
        "binary": {
            "Angle Encoding": 0.8298,
            "Amplitude Encoding": 0.8036,
            "ZZFeatureMap": 0.8820
        },
        "multiclass": {
            "Angle Encoding": 0.7922,
            "Amplitude Encoding": 0.7682,
            "ZZFeatureMap": 0.7477
        }
    }
}

# ============================================================
# SYNTHETIC BINARY SCORES
# ============================================================

def generate_binary_data(n, target_auc, rng):
    """
    Generate synthetic labels and continuous prediction scores.

    The target AUC controls the approximate separation between
    the positive and negative score distributions.
    """

    y_true = rng.integers(0, 2, size=n)

    # AUC of two equally variable normal distributions is
    # approximately Phi(mean_difference / sqrt(2)).
    from scipy.stats import norm

    separation = np.sqrt(2) * norm.ppf(
        np.clip(target_auc, 0.51, 0.99)
    )

    scores = rng.normal(
        loc=(2 * y_true - 1) * separation / 2,
        scale=1.0,
        size=n
    )

    return y_true, scores


# ============================================================
# SYNTHETIC MULTICLASS SCORES
# ============================================================

def generate_multiclass_data(n, n_classes, target_auc, rng):
    """
    Generate six-class labels and synthetic class probabilities.

    Calculates macro-average one-vs-rest ROC-AUC.
    Scores are illustrative, not model predictions.
    """

    from scipy.special import softmax

    y_true = rng.integers(0, n_classes, size=n)
    logits = rng.normal(0, 1, size=(n, n_classes))

    # Increase the score for the correct class to control
    # approximate class separability.
    from scipy.stats import norm

    separation = np.sqrt(2) * norm.ppf(
        np.clip(target_auc, 0.51, 0.99)
    )

    logits[np.arange(n), y_true] += separation

    probabilities = softmax(logits, axis=1)

    return y_true, probabilities


# ============================================================
# PLOT BINARY ROC
# ============================================================

def plot_binary_roc(model_name, curves, filename):
    plt.figure(figsize=(8, 6))

    # Random classifier baseline
    plt.plot(
        [0, 1], [0, 1],
        linestyle="--",
        color="gray",
        label="Random Classifier"
    )

    for curve_name, target_auc in curves.items():
        y_true, scores = generate_binary_data(
            N_BINARY, target_auc, rng
        )

        fpr, tpr, _ = roc_curve(y_true, scores)
        actual_auc = auc(fpr, tpr)

        plt.plot(
            fpr,
            tpr,
            linewidth=2,
            label=f"{curve_name} (AUC = {actual_auc:.3f})"
        )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model_name}: Binary Classification ROC")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.25)

    plt.text(
        0.5, 0.08,
        "SYNTHETIC ILLUSTRATION",
        transform=plt.gca().transAxes,
        fontsize=10,
        alpha=0.7,
        ha="center"
    )

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, filename),
        dpi=300
    )
    plt.close()


# ============================================================
# PLOT MULTICLASS ROC
# Macro-average one-vs-rest ROC
# ============================================================

def plot_multiclass_roc(model_name, curves, filename):
    plt.figure(figsize=(8, 6))

    plt.plot(
        [0, 1], [0, 1],
        linestyle="--",
        color="gray",
        label="Random Classifier"
    )

    for curve_name, target_auc in curves.items():
        y_true, probabilities = generate_multiclass_data(
            N_MULTICLASS,
            N_CLASSES,
            target_auc,
            rng
        )

        y_binary = label_binarize(
            y_true,
            classes=np.arange(N_CLASSES)
        )

        # Compute a one-vs-rest ROC curve for each class.
        class_aucs = []

        for i in range(N_CLASSES):
            fpr, tpr, _ = roc_curve(
                y_binary[:, i],
                probabilities[:, i]
            )

            class_aucs.append(auc(fpr, tpr))

        macro_auc = np.mean(class_aucs)

        # Micro-average ROC curve across all classes.
        fpr_micro, tpr_micro, _ = roc_curve(
            y_binary.ravel(),
            probabilities.ravel()
        )

        plt.plot(
            fpr_micro,
            tpr_micro,
            linewidth=2,
            label=(
                f"{curve_name} "
                f"(Macro AUC = {macro_auc:.3f})"
            )
        )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model_name}: Six-Class ROC")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.25)

    plt.text(
        0.5, 0.08,
        "SYNTHETIC ILLUSTRATION",
        transform=plt.gca().transAxes,
        fontsize=10,
        alpha=0.7,
        ha="center"
    )

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, filename),
        dpi=300
    )
    plt.close()


# ============================================================
# GENERATE ALL 8 GRAPHS
# ============================================================

# 1. Classical ML: Binary
plot_binary_roc(
    "Classical ML",
    {"Classical ML": AUC_TARGETS["Classical ML"]["binary"]},
    "01_classical_binary.png"
)

# 2. Classical ML: Multiclass
plot_multiclass_roc(
    "Classical ML",
    {"Classical ML": AUC_TARGETS["Classical ML"]["multiclass"]},
    "02_classical_multiclass.png"
)

# 3. MLP: Binary
plot_binary_roc(
    "MLP",
    {"MLP": AUC_TARGETS["MLP"]["binary"]},
    "03_mlp_binary.png"
)

# 4. MLP: Multiclass
plot_multiclass_roc(
    "MLP",
    {"MLP": AUC_TARGETS["MLP"]["multiclass"]},
    "04_mlp_multiclass.png"
)

# 5. VQC: Binary, all three encodings
plot_binary_roc(
    "VQC",
    AUC_TARGETS["VQC"]["binary"],
    "05_vqc_binary.png"
)

# 6. VQC: Multiclass, all three encodings
plot_multiclass_roc(
    "VQC",
    AUC_TARGETS["VQC"]["multiclass"],
    "06_vqc_multiclass.png"
)

# 7. QSVM: Binary, all three encodings
plot_binary_roc(
    "QSVM",
    AUC_TARGETS["QSVM"]["binary"],
    "07_qsvm_binary.png"
)

# 8. QSVM: Multiclass, all three encodings
plot_multiclass_roc(
    "QSVM",
    AUC_TARGETS["QSVM"]["multiclass"],
    "08_qsvm_multiclass.png"
)

print("\nAll 8 illustrative ROC graphs generated!")
print("Output directory:", OUTPUT_DIR)