"""Classical ML comparison on the shared 1,000-row PQC experiments.
Run from the directory containing experiment_data/.
"""
from pathlib import Path
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, ConfusionMatrixDisplay,
    roc_curve, auc)

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

SEED = 42
DATA_DIR = Path("experiment_data")
OUT_DIR = DATA_DIR / "classical_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FEATURES = ["e4_entropy_h", "e5_entropy_c", "e6_time_char", "e6b_flow_duration_ms"]


def model_set(task, n_classes):
    models = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED)),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1),
        "Extra Trees": ExtraTreesClassifier(n_estimators=200, random_state=SEED, n_jobs=-1),
    }
    if HAS_XGBOOST:
        if task == "binary":
            models["XGBoost"] = XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.08,
                random_state=SEED, eval_metric="logloss", n_jobs=-1)
        else:
            models["XGBoost"] = XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.08,
                random_state=SEED, eval_metric="mlogloss", objective="multi:softprob",
                num_class=n_classes, n_jobs=-1)
    return models


def run_task(prefix, target_col, task):
    paths = {k: DATA_DIR / f"{prefix}_{suffix}.csv" for k, suffix in {
        "Xtr": "X_train", "Xte": "X_test", "ytr": "y_train", "yte": "y_test"}.items()}
    for p in paths.values():
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}. Run prepare_1000_datasets.py first.")
    Xtr_df, Xte_df = pd.read_csv(paths["Xtr"]), pd.read_csv(paths["Xte"])
    missing = [f for f in FEATURES if f not in Xtr_df.columns or f not in Xte_df.columns]
    if missing:
        raise ValueError(f"Required features missing: {missing}. Available columns: {list(Xtr_df.columns)}")
    Xtr_df, Xte_df = Xtr_df[FEATURES].astype(float), Xte_df[FEATURES].astype(float)
    if Xtr_df.isna().any().any() or Xte_df.isna().any().any():
        raise ValueError("Missing feature values found in the experiment CSVs.")
    ytr_raw = pd.read_csv(paths["ytr"])[target_col]
    yte_raw = pd.read_csv(paths["yte"])[target_col]
    if task == "binary":
        ytr, yte = ytr_raw.astype(int).to_numpy(), yte_raw.astype(int).to_numpy()
        class_names = ["Non-PQC", "PQC"]
    else:
        enc = LabelEncoder()
        ytr = enc.fit_transform(ytr_raw.astype(str))
        yte = enc.transform(yte_raw.astype(str))
        class_names = list(enc.classes_)
        joblib.dump(enc, OUT_DIR / f"{prefix}_label_encoder.joblib")
    # Fit scaling on training data only. Pipelines that need scaling fit it inside model.fit.
    scaler = StandardScaler().fit(Xtr_df)
    Xtr, Xte = scaler.transform(Xtr_df), scaler.transform(Xte_df)
    joblib.dump(scaler, OUT_DIR / f"{prefix}_scaler.joblib")
    all_results = []
    roc_fig, roc_ax = plt.subplots(figsize=(8, 6)) if task == "binary" else (None, None)
    for name, model in model_set(task, len(class_names)).items():
        print(f"\n{'='*65}\n{task.upper()} | {name}\n{'='*65}")
        start = time.perf_counter(); model.fit(Xtr, ytr); train_time = time.perf_counter() - start
        start = time.perf_counter(); pred = model.predict(Xte); infer_time = time.perf_counter() - start
        probs = model.predict_proba(Xte) if hasattr(model, "predict_proba") else None
        avg = "binary" if task == "binary" else "macro"
        acc = accuracy_score(yte, pred)
        prec = precision_score(yte, pred, average=avg, zero_division=0)
        rec = recall_score(yte, pred, average=avg, zero_division=0)
        f1 = f1_score(yte, pred, average=avg, zero_division=0)
        auc_score = np.nan
        if probs is not None:
            try:
                if task == "binary": auc_score = roc_auc_score(yte, probs[:, 1])
                else: auc_score = roc_auc_score(yte, probs, multi_class="ovr", average="macro", labels=np.arange(len(class_names)))
            except ValueError as e:
                print(f"ROC-AUC unavailable: {e}")
        cm = confusion_matrix(yte, pred, labels=np.arange(len(class_names)))
        print(f"Accuracy: {acc*100:.2f}% | Precision: {prec*100:.2f}% | Recall: {rec*100:.2f}% | F1: {f1*100:.2f}%")
        print(f"ROC-AUC: {auc_score:.4f} | Train: {train_time:.3f}s | Inference: {infer_time:.4f}s")
        print("Confusion matrix:\n", cm)
        print(classification_report(yte, pred, labels=np.arange(len(class_names)), target_names=class_names, zero_division=0))
        all_results.append({"Task": task, "Model": name, "Accuracy (%)": acc*100, "Precision (%)": prec*100,
            "Recall (%)": rec*100, "F1 (%)": f1*100, "ROC-AUC": auc_score,
            "Training Time (s)": train_time, "Inference Time (s)": infer_time})
        fig, ax = plt.subplots(figsize=(7, 6))
        ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax, xticks_rotation=45, colorbar=False)
        ax.set_title(f"{task.title()} | {name} confusion matrix"); fig.tight_layout()
        fig.savefig(OUT_DIR / f"{prefix}_{name.lower().replace(' ', '_')}_confusion_matrix.png", dpi=180); plt.close(fig)
        if task == "binary" and probs is not None:
            fpr, tpr, _ = roc_curve(yte, probs[:, 1]); roc_ax.plot(fpr, tpr, label=f"{name} (AUC={auc_score:.3f})")
        joblib.dump(model, OUT_DIR / f"{prefix}_{name.lower().replace(' ', '_')}.joblib")
    if task == "binary":
        roc_ax.plot([0,1],[0,1], "--", label="Random classifier")
        roc_ax.set(xlabel="False positive rate", ylabel="True positive rate", title="Binary PQC detection ROC", xlim=(0,1), ylim=(0,1.02))
        roc_ax.grid(True, alpha=0.3); roc_ax.legend(); roc_fig.tight_layout()
        roc_fig.savefig(OUT_DIR / "binary_combined_roc.png", dpi=200); plt.close(roc_fig)
    df = pd.DataFrame(all_results)
    df.to_csv(OUT_DIR / f"{prefix}_classical_comparison.csv", index=False)
    print("\nSUMMARY\n", df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return all_results

if __name__ == "__main__":
    results = []
    results += run_task("binary", "PQC_label", "binary")
    results += run_task("multiclass", "Crypto_configuration", "multiclass")
    pd.DataFrame(results).to_csv(OUT_DIR / "all_classical_results.csv", index=False)
    print(f"\nAll classical results saved to: {OUT_DIR.resolve()}")
