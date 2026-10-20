
import pandas as pd
import time
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

# ============================================================
# 1. LOAD DATA
# ============================================================

X_train = pd.read_csv("KEM_X_train_scaled.csv")
X_test = pd.read_csv("KEM_X_test_scaled.csv")

y_train = pd.read_csv("KEM_y_train.csv")["KEM_label"]
y_test = pd.read_csv("KEM_y_test.csv")["KEM_label"]

# ============================================================
# 2. KEM ABLATION — REMOVE SIZE FEATURES
# ============================================================

remove_features = [
    "e2c_total_bytes",
    "e2_client_size",
    "e2_client_record_len",
    "e2_server_record_len"
]

X_train = X_train.drop(columns=remove_features)
X_test = X_test.drop(columns=remove_features)

print("\nFeatures used after ablation:")
print(list(X_train.columns))

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)

print("\nClass mapping:")
print("0 = ML-KEM-768")
print("1 = ML-KEM-1024")

# ============================================================
# 3. DEFINE MODELS
# ============================================================

models = {
    "Logistic Regression": LogisticRegression(
        random_state=42,
        max_iter=1000
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
        max_depth=10
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1
    )
}

# ============================================================
# 4. TRAIN, EVALUATE AND COLLECT ROC DATA
# ============================================================

results = []

plt.figure(figsize=(9, 7))

for name, model in models.items():

    print("\n" + "=" * 60)
    print(f"{name.upper()} — ML-KEM TYPE CLASSIFICATION")
    print("=" * 60)

    # Training time
    start_time = time.perf_counter()
    model.fit(X_train, y_train)
    training_time = time.perf_counter() - start_time

    # Predictions
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]

    # Evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, zero_division=0
    )
    recall = recall_score(
        y_test, y_pred, zero_division=0
    )
    f1 = f1_score(
        y_test, y_pred, zero_division=0
    )
    auc_value = roc_auc_score(y_test, y_score)

    cm = confusion_matrix(y_test, y_pred)

    print(f"\nAccuracy      : {accuracy * 100:.2f}%")
    print(f"Precision     : {precision * 100:.2f}%")
    print(f"Recall        : {recall * 100:.2f}%")
    print(f"F1-score      : {f1 * 100:.2f}%")
    print(f"ROC-AUC       : {auc_value * 100:.2f}%")
    print(f"Training time : {training_time:.4f} seconds")

    print("\nConfusion Matrix:")
    print(cm)

    # Store metrics
    results.append({
        "Model": name,
        "Accuracy (%)": accuracy * 100,
        "Precision (%)": precision * 100,
        "Recall (%)": recall * 100,
        "F1-score (%)": f1 * 100,
        "ROC-AUC (%)": auc_value * 100,
        "Training Time (s)": training_time
    })

    # Calculate and plot ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_score)

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"{name} (AUC = {auc_value:.3f})"
    )

# ============================================================
# 5. ROC CURVE
# ============================================================

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=1.5,
    label="Random Classifier (AUC = 0.500)"
)

plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)

plt.title(
    "ROC Curves: ML-KEM-768 vs ML-KEM-1024",
    fontsize=14
)

plt.xlim(0, 1)
plt.ylim(0, 1.02)
plt.legend(loc="lower right", fontsize=9)
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    "KEM_ROC_curves.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nROC curve saved as KEM_ROC_curves.png")

# ============================================================
# 6. FINAL COMPARISON TABLE
# ============================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 90)
print("FINAL ML-KEM CLASSIFICATION COMPARISON")
print("=" * 90)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)

# ============================================================
# 7. SAVE RESULTS
# ============================================================

results_df.to_csv(
    "KEM_classical_ML_results.csv",
    index=False
)

print("\nResults saved to: KEM_classical_ML_results.csv")
