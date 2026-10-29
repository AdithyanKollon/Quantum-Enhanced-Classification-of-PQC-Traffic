"""Run MLP on both experiment_data splits created by prepare_1000_datasets.py."""
from pathlib import Path
import time
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, ConfusionMatrixDisplay, roc_curve, auc
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Sequential

SEED = 42
DATA_DIR = Path("experiment_data")
OUT_DIR = DATA_DIR / "mlp_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def make_model(n_features, task, n_classes=2):
    model = Sequential([
        Input(shape=(n_features,)),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dropout(0.2),
        Dense(16, activation="relu"),
    ])
    if task == "binary":
        model.add(Dense(1, activation="sigmoid"))
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    else:
        model.add(Dense(n_classes, activation="softmax"))
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def run_one(prefix, target_col, task):
    paths = {
        "Xtr": DATA_DIR / f"{prefix}_X_train.csv",
        "Xte": DATA_DIR / f"{prefix}_X_test.csv",
        "ytr": DATA_DIR / f"{prefix}_y_train.csv",
        "yte": DATA_DIR / f"{prefix}_y_test.csv",
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Run prepare_1000_datasets.py first.")

    Xtr_df, Xte_df = pd.read_csv(paths["Xtr"]), pd.read_csv(paths["Xte"])
    ytr_raw = pd.read_csv(paths["ytr"])[target_col]
    yte_raw = pd.read_csv(paths["yte"])[target_col]
    if list(Xtr_df.columns) != list(Xte_df.columns):
        raise ValueError(f"{prefix}: train/test feature columns differ.")
    if Xtr_df.isna().any().any() or Xte_df.isna().any().any():
        raise ValueError(f"{prefix}: missing feature values detected.")

    # Fit preprocessing only on the training partition.
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr_df.astype(float))
    Xte = scaler.transform(Xte_df.astype(float))
    joblib.dump(scaler, OUT_DIR / f"{prefix}_scaler.joblib")

    if task == "binary":
        ytr = ytr_raw.astype(int).to_numpy()
        yte = yte_raw.astype(int).to_numpy()
        if not set(np.unique(ytr)).issubset({0, 1}) or len(np.unique(ytr)) != 2:
            raise ValueError(f"{prefix}: expected binary labels 0 and 1.")
        names = ["Non-PQC", "PQC"]
        n_classes = 2
    else:
        encoder = LabelEncoder()
        ytr = encoder.fit_transform(ytr_raw.astype(str))
        try:
            yte = encoder.transform(yte_raw.astype(str))
        except ValueError as exc:
            raise ValueError("A test class is absent from training labels.") from exc
        names = list(encoder.classes_)
        n_classes = len(names)
        if n_classes != 6:
            warnings.warn(f"Expected six classes, found {n_classes}: {names}")
        joblib.dump(encoder, OUT_DIR / f"{prefix}_label_encoder.joblib")

    # Stratified validation subset is drawn only from the training split.
    Xfit, Xval, yfit, yval = train_test_split(
        Xtr, ytr, test_size=0.2, random_state=SEED, stratify=ytr
    )
    model = make_model(Xtr.shape[1], task, n_classes)
    stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    start = time.perf_counter()
    history = model.fit(
        Xfit, yfit, validation_data=(Xval, yval), epochs=100, batch_size=32,
        callbacks=[stop], verbose=1
    )
    train_seconds = time.perf_counter() - start
    start = time.perf_counter()
    probs = model.predict(Xte, verbose=0)
    infer_seconds = time.perf_counter() - start

    if task == "binary":
        scores = probs.reshape(-1)
        pred = (scores >= 0.5).astype(int)
        precision = precision_score(yte, pred, zero_division=0)
        recall = recall_score(yte, pred, zero_division=0)
        f1 = f1_score(yte, pred, zero_division=0)
        roc_auc = roc_auc_score(yte, scores)
        fpr, tpr, _ = roc_curve(yte, scores)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(fpr, tpr, label=f"MLP (AUC={roc_auc:.3f})")
        ax.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")
        ax.set(xlabel="False positive rate", ylabel="True positive rate", title="Binary PQC detection ROC")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"{prefix}_roc.png", dpi=300)
        plt.close(fig)
        labels = [0, 1]
        display_names = names
    else:
        pred = np.argmax(probs, axis=1)
        precision = precision_score(yte, pred, average="macro", zero_division=0)
        recall = recall_score(yte, pred, average="macro", zero_division=0)
        f1 = f1_score(yte, pred, average="macro", zero_division=0)
        labels = list(range(n_classes))
        display_names = names
        yte_bin = label_binarize(yte, classes=labels)
        try:
            roc_auc = roc_auc_score(yte_bin, probs, average="macro", multi_class="ovr")
            fig, ax = plt.subplots(figsize=(8, 6))
            for i, class_name in enumerate(names):
                fpr, tpr, _ = roc_curve(yte_bin[:, i], probs[:, i])
                ax.plot(fpr, tpr, label=f"{class_name} (AUC={auc(fpr, tpr):.3f})")
            ax.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")
            ax.set(xlabel="False positive rate", ylabel="True positive rate", title="Six-class PQC configuration ROC")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(OUT_DIR / f"{prefix}_roc_ovr.png", dpi=300)
            plt.close(fig)
        except ValueError:
            roc_auc = np.nan
            print("ROC-AUC unavailable for this test split.")

    accuracy = accuracy_score(yte, pred)
    cm = confusion_matrix(yte, pred, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay(cm, display_labels=display_names).plot(
        ax=ax, xticks_rotation=45, colorbar=False
    )
    ax.set_title(f"{prefix}: MLP confusion matrix")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{prefix}_confusion_matrix.png", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history.history["loss"], label="Training")
    axes[0].plot(history.history["val_loss"], label="Validation")
    axes[0].set(title="Loss", xlabel="Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(history.history["accuracy"], label="Training")
    axes[1].plot(history.history["val_accuracy"], label="Validation")
    axes[1].set(title="Accuracy", xlabel="Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{prefix}_training_history.png", dpi=300)
    plt.close(fig)

    model.save(OUT_DIR / f"{prefix}_mlp.keras")
    report = classification_report(
        yte, pred, labels=labels, target_names=display_names,
        output_dict=True, zero_division=0
    )
    pd.DataFrame(report).transpose().to_csv(OUT_DIR / f"{prefix}_classification_report.csv")

    result = {
        "Experiment": prefix, "Model": "MLP", "Training samples": len(ytr),
        "Testing samples": len(yte), "Features": Xtr.shape[1],
        "Accuracy": accuracy, "Precision": precision, "Recall": recall,
        "F1": f1, "ROC_AUC": roc_auc, "Training time (s)": train_seconds,
        "Inference time (s)": infer_seconds, "Epochs": len(history.history["loss"])
    }
    print(f"\n{prefix} results")
    print("Classes:", display_names)
    for key in ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]:
        print(f"{key}: {result[key]:.4f}")
    print(f"Training: {train_seconds:.2f}s | Inference: {infer_seconds:.4f}s")
    print("Confusion matrix:\n", cm)
    return result


def main():
    results = [
        run_one("binary", "PQC_label", "binary"),
        run_one("multiclass", "Crypto_configuration", "multiclass"),
    ]
    summary = pd.DataFrame(results)
    summary.to_csv(OUT_DIR / "mlp_summary.csv", index=False)
    print("\nSummary:\n", summary.to_string(index=False))
    print(f"\nOutputs saved in: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
