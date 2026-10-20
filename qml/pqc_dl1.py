
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

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

X_train = pd.read_csv("PQC_X_train_scaled.csv")
X_test = pd.read_csv("PQC_X_test_scaled.csv")

y_train = pd.read_csv("PQC_y_train.csv").squeeze()
y_test = pd.read_csv("PQC_y_test.csv").squeeze()

# ============================================================
# 2. FEATURE ABLATION
# ============================================================

remove_features = [
    "e2c_total_bytes",
    "e2_client_size",
    "e2_client_record_len",
    "e2_server_record_len"
]

X_train = X_train.drop(columns=remove_features)
X_test = X_test.drop(columns=remove_features)

print("Features used:")
print(list(X_train.columns))
print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)

# ============================================================
# 3. BUILD MLP MODEL
# ============================================================

model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(64, activation="relu"),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dropout(0.2),
    Dense(16, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ============================================================
# 4. EARLY STOPPING
# ============================================================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

# ============================================================
# 5. TRAIN MODEL
# ============================================================

start_time = time.perf_counter()

history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

training_time = time.perf_counter() - start_time

# ============================================================
# 6. PREDICTION
# ============================================================

y_prob = model.predict(X_test, verbose=0).ravel()
y_pred = (y_prob >= 0.5).astype(int)

# ============================================================
# 7. EVALUATION METRICS
# ============================================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_prob)

cm = confusion_matrix(y_test, y_pred)

print("\n" + "=" * 80)
print("DEEP LEARNING — PQC vs NON-PQC CLASSIFICATION")
print("=" * 80)

print("Classes: 0 = Non-PQC, 1 = PQC")
print(f"Accuracy     : {accuracy * 100:.2f}%")
print(f"Precision    : {precision * 100:.2f}%")
print(f"Recall       : {recall * 100:.2f}%")
print(f"F1-score     : {f1 * 100:.2f}%")
print(f"ROC-AUC      : {roc_auc * 100:.2f}%")
print(f"Training time: {training_time:.2f} seconds")
print(f"Epochs run   : {len(history.history['loss'])}")

print("\nConfusion Matrix:")
print(cm)

# ============================================================
# 8. PLOT ROC CURVE
# ============================================================

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    linewidth=2,
    label=f"MLP (AUC = {roc_auc:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=1.5,
    label="Random Classifier (AUC = 0.500)"
)

plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.title("ROC Curve — MLP for PQC Detection", fontsize=14)

plt.xlim(0, 1)
plt.ylim(0, 1.02)
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    "PQC_MLP_ROC_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nROC curve saved as PQC_MLP_ROC_curve.png")

# ============================================================
# 9. PLOT TRAINING HISTORY
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

axes[0].plot(history.history["loss"], label="Training Loss")
axes[0].plot(history.history["val_loss"], label="Validation Loss")
axes[0].set_title("MLP Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Binary Cross-Entropy Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history["accuracy"], label="Training Accuracy")
axes[1].plot(history.history["val_accuracy"], label="Validation Accuracy")
axes[1].set_title("MLP Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    "PQC_MLP_training_history.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Training history saved as PQC_MLP_training_history.png")

# ============================================================
# 10. SAVE METRICS
# ============================================================

results_df = pd.DataFrame([{
    "Model": "MLP",
    "Accuracy (%)": accuracy * 100,
    "Precision (%)": precision * 100,
    "Recall (%)": recall * 100,
    "F1-score (%)": f1 * 100,
    "ROC-AUC (%)": roc_auc * 100,
    "Training Time (s)": training_time,
    "Epochs": len(history.history["loss"])
}])

results_df.to_csv("PQC_MLP_results.csv", index=False)

print("\nResults saved to PQC_MLP_results.csv")
