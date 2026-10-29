"""
Generate 8 ROC-AUC plots for the PQC traffic classification project:
  1. Classical ML - binary
  2. Classical ML - multiclass
  3. MLP - binary
  4. MLP - multiclass
  5. VQC - binary
  6. VQC - multiclass
  7. QSVM - binary
  8. QSVM - multiclass

IMPORTANT:
- Uses saved experiment_data train/test splits.
- Uses actual model prediction scores if available.
- For classical ML and MLP, this script can retrain the models using the saved split.
- For VQC/QSVM, provide saved test prediction scores/probabilities from your QML scripts.
- It will NOT invent ROC curves from accuracy/AUC summary values alone.
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.neural_network import MLPClassifier

# ------------------------- CONFIGURATION -------------------------
DATA_DIR = Path("experiment_data")
OUT_DIR = DATA_DIR / "roc_auc_curves"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "e4_entropy_h",
    "e5_entropy_c",
    "e6_time_char",
    "e6b_flow_duration_ms",
]

# Change these names ONLY if your saved split files use different filenames.
SPLIT_FILES = {
    "binary": {
        "train": DATA_DIR / "binary_train.csv",
        "test": DATA_DIR / "binary_test.csv",
        "target": "PQC_label",
    },
    "multiclass": {
        "train": DATA_DIR / "multiclass_train.csv",
        "test": DATA_DIR / "multiclass_test.csv",
        "target": "Crypto_configuration",
    },
}

# Prediction-score files expected for QML.
# Each file should contain one row per test sample, in the same order as the
# corresponding saved test CSV.
#
# Binary:
#   y_score column = predicted probability/decision score for positive class (PQC=1)
# Multiclass:
#   one column per class, named exactly as the class labels in the test CSV.
#
# You can edit these paths to match files produced by your QML scripts.
QML_SCORE_FILES = {
    ("VQC", "binary"): DATA_DIR / "vqc_3_encoding_results" / "binary_angle_test_scores.csv",
    ("VQC", "multiclass"): DATA_DIR / "vqc_3_encoding_results" / "multiclass_angle_test_scores.csv",
    ("QSVM", "binary"): DATA_DIR / "qsvm_3_encoding_results" / "binary_angle_test_scores.csv",
    ("QSVM", "multiclass"): DATA_DIR / "qsvm_3_encoding_results" / "multiclass_angle_test_scores.csv",
}

# ------------------------- HELPERS -------------------------
def load_split(task):
    cfg = SPLIT_FILES[task]
    if not cfg["train"].exists() or not cfg["test"].exists():
        raise FileNotFoundError(
            f"Missing split for {task}. Expected:\n"
            f"  {cfg['train']}\n  {cfg['test']}\n"
            "Edit SPLIT_FILES near the top of this script to match your filenames."
        )

    train = pd.read_csv(cfg["train"])
    test = pd.read_csv(cfg["test"])
    target = cfg["target"]

    missing = [c for c in FEATURES + [target] if c not in train.columns or c not in test.columns]
    if missing:
        raise ValueError(
            f"Missing columns for {task}: {missing}\n"
            f"Columns in train: {list(train.columns)}"
        )

    X_train = train[FEATURES].apply(pd.to_numeric, errors="coerce")
    X_test = test[FEATURES].apply(pd.to_numeric, errors="coerce")
    if X_train.isna().any().any() or X_test.isna().any().any():
        raise ValueError("Feature columns contain missing/non-numeric values. Clean/impute them consistently first.")

    y_train = train[target].astype(str) if task == "multiclass" else train[target].astype(int)
    y_test = test[target].astype(str) if task == "multiclass" else test[target].astype(int)

    # Fit scaling only on training data, then transform test data.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test


def get_classical_model(name):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=3000, class_weight=None),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
        "Extra Trees": ExtraTreesClassifier(n_estimators=300, random_state=42),
    }
    if name not in models:
        raise ValueError(f"Unknown model: {name}")
    return models[name]


def positive_scores(model, X_test):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)
    if hasattr(model, "decision_function"):
        return model.decision_function(X_test)
    raise ValueError(f"{type(model).__name__} has neither predict_proba nor decision_function.")


def plot_binary(y_true, scores, title, out_path):
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores)
    if scores.ndim == 2:
        if scores.shape[1] == 2:
            scores = scores[:, 1]
        else:
            raise ValueError("Binary scores must be a 1D positive-class score or 2-column probability array.")
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 7))
    plt.plot(fpr, tpr, linewidth=2.5, label=f"{title} (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=2, label="Random classifier")
    plt.xlim(0, 1)
    plt.ylim(0, 1.02)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Binary PQC detection ROC")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path} | AUC={roc_auc:.4f}")


def plot_multiclass(y_true, scores, classes, title, out_path):
    y_true = np.asarray(y_true).astype(str)
    classes = [str(c) for c in classes]
    scores = np.asarray(scores, dtype=float)

    if scores.ndim != 2 or scores.shape[1] != len(classes):
        raise ValueError(
            f"Multiclass score array must have shape (n_test, {len(classes)}), "
            f"but got {scores.shape}."
        )

    y_bin = label_binarize(y_true, classes=classes)
    plt.figure(figsize=(12, 8))

    # One-vs-rest ROC for each class
    for i, cls in enumerate(classes):
        if len(np.unique(y_bin[:, i])) < 2:
            warnings.warn(f"Skipping class {cls}: test set does not contain both binary outcomes.")
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], scores[:, i])
        class_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, linewidth=1.8, label=f"{cls} (AUC={class_auc:.3f})")

    # Micro-average curve
    fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), scores.ravel())
    micro_auc = auc(fpr_micro, tpr_micro)
    plt.plot(
        fpr_micro, tpr_micro, linestyle=":", linewidth=3,
        label=f"Micro-average (AUC={micro_auc:.3f})"
    )
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=2, label="Random classifier")
    plt.xlim(0, 1)
    plt.ylim(0, 1.02)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Multiclass PQC configuration ROC (one-vs-rest)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path} | micro-AUC={micro_auc:.4f}")


def load_qml_scores(path, task, y_test):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing QML prediction-score file: {path}\n"
            "The summary table contains metrics only, not per-test prediction scores. "
            "Export actual test probabilities/scores from the QSVM/VQC script; "
            "do not reconstruct ROC curves from a single AUC/accuracy value."
        )
    df = pd.read_csv(path)
    if task == "binary":
        if "y_score" in df.columns:
            return df["y_score"].to_numpy()
        if "prob_PQC" in df.columns:
            return df["prob_PQC"].to_numpy()
        if "score" in df.columns:
            return df["score"].to_numpy()
        raise ValueError(f"{path} needs a y_score, prob_PQC, or score column.")
    classes = sorted(pd.Series(y_test).astype(str).unique().tolist())
    missing = [c for c in classes if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing probability columns for classes: {missing}")
    return df[classes].to_numpy()


# ------------------------- CLASSICAL ROC PLOTS -------------------------
def run_classical(task):
    X_train, X_test, y_train, y_test = load_split(task)

    if task == "binary":
        # Use Extra Trees, the strongest binary classical model in the supplied results.
        model = ExtraTreesClassifier(n_estimators=300, random_state=42)
        model.fit(X_train, y_train)
        scores = positive_scores(model, X_test)
        plot_binary(y_test, scores, "Extra Trees", OUT_DIR / "01_classical_binary_roc.png")
    else:
        # XGBoost was strongest in the supplied multiclass results.
        try:
            from xgboost import XGBClassifier
            class_names = sorted(pd.Series(y_train).astype(str).unique().tolist())
            class_to_id = {name: i for i, name in enumerate(class_names)}
            y_train_ids = pd.Series(y_train).map(class_to_id).to_numpy()
            model = XGBClassifier(
                objective="multi:softprob",
                num_class=len(class_names),
                n_estimators=250,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="mlogloss",
                random_state=42,
            )
            model.fit(X_train, y_train_ids)
            scores = model.predict_proba(X_test)
            plot_multiclass(
                y_test, scores, class_names, "XGBoost",
                OUT_DIR / "02_classical_multiclass_roc.png"
            )
        except ImportError:
            print("xgboost not installed. Using Extra Trees for the multiclass classical ROC instead.")
            model = ExtraTreesClassifier(n_estimators=300, random_state=42)
            model.fit(X_train, y_train)
            scores = model.predict_proba(X_test)
            classes = [str(c) for c in model.classes_]
            plot_multiclass(
                y_test, scores, classes, "Extra Trees",
                OUT_DIR / "02_classical_multiclass_roc.png"
            )


# ------------------------- MLP ROC PLOTS -------------------------
def run_mlp(task):
    X_train, X_test, y_train, y_test = load_split(task)

    if task == "binary":
        model = MLPClassifier(
            hidden_layer_sizes=(64, 32), activation="relu",
            solver="adam", max_iter=1000, random_state=42,
            early_stopping=True, n_iter_no_change=20
        )
        model.fit(X_train, y_train)
        scores = model.predict_proba(X_test)[:, 1]
        plot_binary(y_test, scores, "MLP", OUT_DIR / "03_mlp_binary_roc.png")
    else:
        classes = sorted(pd.Series(y_train).astype(str).unique().tolist())
        model = MLPClassifier(
            hidden_layer_sizes=(64, 32), activation="relu",
            solver="adam", max_iter=1000, random_state=42,
            early_stopping=True, n_iter_no_change=20
        )
        model.fit(X_train, y_train.astype(str))
        raw = model.predict_proba(X_test)
        model_classes = [str(c) for c in model.classes_]
        # Ensure columns match the same sorted class order used for binarization.
        scores = raw[:, [model_classes.index(c) for c in classes]]
        plot_multiclass(
            y_test, scores, classes, "MLP",
            OUT_DIR / "04_mlp_multiclass_roc.png"
        )


# ------------------------- QML ROC PLOTS -------------------------
def run_qml(model_name, task, out_name):
    _, _, _, y_test = load_split(task)
    score_path = QML_SCORE_FILES[(model_name, task)]
    scores = load_qml_scores(score_path, task, y_test)

    if task == "binary":
        plot_binary(
            y_test, scores, f"{model_name} (Angle Encoding)",
            OUT_DIR / out_name
        )
    else:
        classes = sorted(pd.Series(y_test).astype(str).unique().tolist())
        plot_multiclass(
            y_test, scores, classes, f"{model_name} (Angle Encoding)",
            OUT_DIR / out_name
        )


if __name__ == "__main__":
    # 1-2: Classical ML
    run_classical("binary")
    run_classical("multiclass")

    # 3-4: MLP
    run_mlp("binary")
    run_mlp("multiclass")

    # 5-8: QML. These require real saved test scores from the QML runs.
    run_qml("VQC", "binary", "05_vqc_binary_roc.png")
    run_qml("VQC", "multiclass", "06_vqc_multiclass_roc.png")
    run_qml("QSVM", "binary", "07_qsvm_binary_roc.png")
    run_qml("QSVM", "multiclass", "08_qsvm_multiclass_roc.png")

    print(f"\nAll ROC plots saved in: {OUT_DIR.resolve()}")
