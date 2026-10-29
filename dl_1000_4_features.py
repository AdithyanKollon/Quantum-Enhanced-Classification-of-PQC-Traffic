"""TensorFlow MLP / deep-learning comparison on the shared 1,000-row experiments.
Run from the directory containing experiment_data/.
"""
from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, ConfusionMatrixDisplay, roc_curve)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

SEED = 42
np.random.seed(SEED); tf.random.set_seed(SEED)
DATA_DIR = Path("experiment_data")
OUT_DIR = DATA_DIR / "dl_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FEATURES = ["e4_entropy_h", "e5_entropy_c", "e6_time_char", "e6b_flow_duration_ms"]


def build_model(n_features, task, n_classes):
    model = Sequential([Input(shape=(n_features,)), Dense(64, activation="relu"), Dropout(0.2),
        Dense(32, activation="relu"), Dropout(0.2), Dense(16, activation="relu")])
    if task == "binary":
        model.add(Dense(1, activation="sigmoid"))
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    else:
        model.add(Dense(n_classes, activation="softmax"))
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def run_task(prefix, target_col, task):
    paths = {k: DATA_DIR / f"{prefix}_{suffix}.csv" for k, suffix in {
        "Xtr": "X_train", "Xte": "X_test", "ytr": "y_train", "yte": "y_test"}.items()}
    for p in paths.values():
        if not p.exists(): raise FileNotFoundError(f"Missing {p}. Run prepare_1000_datasets.py first.")
    Xtr_df, Xte_df = pd.read_csv(paths["Xtr"]), pd.read_csv(paths["Xte"])
    missing = [f for f in FEATURES if f not in Xtr_df.columns or f not in Xte_df.columns]
    if missing: raise ValueError(f"Required features missing: {missing}. Available columns: {list(Xtr_df.columns)}")
    Xtr_df, Xte_df = Xtr_df[FEATURES].astype(float), Xte_df[FEATURES].astype(float)
    if Xtr_df.isna().any().any() or Xte_df.isna().any().any(): raise ValueError("Missing feature values found.")
    ytr_raw = pd.read_csv(paths["ytr"])[target_col]
    yte_raw = pd.read_csv(paths["yte"])[target_col]
    if task == "binary":
        ytr, yte = ytr_raw.astype(int).to_numpy(), yte_raw.astype(int).to_numpy()
        n_classes = 2; class_names = ["Non-PQC", "PQC"]
    else:
        enc = LabelEncoder(); ytr = enc.fit_transform(ytr_raw.astype(str)); yte = enc.transform(yte_raw.astype(str))
        n_classes = len(enc.classes_); class_names = list(enc.classes_)
        joblib.dump(enc, OUT_DIR / f"{prefix}_label_encoder.joblib")
    scaler = StandardScaler().fit(Xtr_df)
    Xtr_all, Xte = scaler.transform(Xtr_df), scaler.transform(Xte_df)
    joblib.dump(scaler, OUT_DIR / f"{prefix}_scaler.joblib")
    # Validation data is taken only from the training partition, preserving the fixed test set.
    Xfit, Xval, yfit, yval = train_test_split(Xtr_all, ytr, test_size=0.2, random_state=SEED, stratify=ytr)
    model = build_model(len(FEATURES), task, n_classes)
    model.summary()
    early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    start = time.perf_counter()
    history = model.fit(Xfit, yfit, validation_data=(Xval, yval), epochs=100, batch_size=32,
        callbacks=[early_stop], verbose=1)
    train_time = time.perf_counter() - start
    start = time.perf_counter(); probs = model.predict(Xte, verbose=0); infer_time = time.perf_counter() - start
    if task == "binary":
        scores = probs.ravel(); pred = (scores >= 0.5).astype(int)
        auc_value = roc_auc_score(yte, scores)
    else:
        pred = np.argmax(probs, axis=1)
        try: auc_value = roc_auc_score(yte, probs, multi_class="ovr", average="macro", labels=np.arange(n_classes))
        except ValueError as exc: print("ROC-AUC unavailable:", exc); auc_value = np.nan
    avg = "binary" if task == "binary" else "macro"
    acc = accuracy_score(yte, pred); prec = precision_score(yte, pred, average=avg, zero_division=0)
    rec = recall_score(yte, pred, average=avg, zero_division=0); f1 = f1_score(yte, pred, average=avg, zero_division=0)
    cm = confusion_matrix(yte, pred, labels=np.arange(n_classes))
    print(f"\n{'='*65}\nDEEP LEARNING MLP | {task.upper()}\n{'='*65}")
    print(f"Classes: {class_names}\nFeatures: {FEATURES}\nTrain samples: {len(ytr)} | Test samples: {len(yte)}")
    print(f"Accuracy: {acc*100:.2f}% | Precision: {prec*100:.2f}% | Recall: {rec*100:.2f}% | F1: {f1*100:.2f}%")
    print(f"ROC-AUC: {auc_value:.4f} | Training: {train_time:.3f}s | Inference: {infer_time:.4f}s | Epochs: {len(history.history['loss'])}")
    print("Confusion matrix:\n", cm)
    print(classification_report(yte, pred, labels=np.arange(n_classes), target_names=class_names, zero_division=0))
    pd.DataFrame([{"Task": task, "Model": "MLP", "Accuracy (%)": acc*100, "Precision (%)": prec*100,
        "Recall (%)": rec*100, "F1 (%)": f1*100, "ROC-AUC": auc_value, "Training Time (s)": train_time,
        "Inference Time (s)": infer_time, "Epochs": len(history.history["loss"])}]).to_csv(OUT_DIR / f"{prefix}_dl_metrics.csv", index=False)
    fig, ax = plt.subplots(figsize=(7,6)); ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax, xticks_rotation=45, colorbar=False)
    ax.set_title(f"{task.title()} | MLP confusion matrix"); fig.tight_layout(); fig.savefig(OUT_DIR / f"{prefix}_mlp_confusion_matrix.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7,5)); ax.plot(history.history["loss"], label="Training loss"); ax.plot(history.history["val_loss"], label="Validation loss")
    ax.set(xlabel="Epoch", ylabel="Loss", title=f"{task.title()} | MLP learning curve"); ax.legend(); ax.grid(True, alpha=0.3); fig.tight_layout()
    fig.savefig(OUT_DIR / f"{prefix}_mlp_learning_curve.png", dpi=180); plt.close(fig)
    if task == "binary":
        fpr, tpr, _ = roc_curve(yte, scores); fig, ax = plt.subplots(figsize=(7,5))
        ax.plot(fpr, tpr, label=f"MLP (AUC={auc_value:.3f})"); ax.plot([0,1],[0,1], "--", label="Random classifier")
        ax.set(xlabel="False positive rate", ylabel="True positive rate", title="Binary PQC detection ROC", xlim=(0,1), ylim=(0,1.02))
        ax.grid(True, alpha=0.3); ax.legend(); fig.tight_layout(); fig.savefig(OUT_DIR / "binary_mlp_roc.png", dpi=200); plt.close(fig)
    model.save(OUT_DIR / f"{prefix}_mlp.keras")
    return {"Task": task, "Model": "MLP", "Accuracy (%)": acc*100, "Precision (%)": prec*100, "Recall (%)": rec*100,
        "F1 (%)": f1*100, "ROC-AUC": auc_value, "Training Time (s)": train_time, "Inference Time (s)": infer_time,
        "Epochs": len(history.history["loss"])}

if __name__ == "__main__":
    rows = [run_task("binary", "PQC_label", "binary"), run_task("multiclass", "Crypto_configuration", "multiclass")]
    pd.DataFrame(rows).to_csv(OUT_DIR / "all_dl_results.csv", index=False)
    print(f"\nAll DL results saved to: {OUT_DIR.resolve()}")
