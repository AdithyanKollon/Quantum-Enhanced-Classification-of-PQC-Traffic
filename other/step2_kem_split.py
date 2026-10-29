import pandas as pd
from sklearn.model_selection import train_test_split

# Load data
X = pd.read_csv("KEM_X_features.csv")
y = pd.read_csv("KEM_y_target.csv")["KEM_label"]

# 80/20 stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Save
X_train.to_csv("KEM_X_train.csv", index=False)
X_test.to_csv("KEM_X_test.csv", index=False)

y_train.to_csv("KEM_y_train.csv", index=False, header=["KEM_label"])
y_test.to_csv("KEM_y_test.csv", index=False, header=["KEM_label"])

print("=" * 60)
print("KEM TRAIN/TEST SPLIT")
print("=" * 60)

print("\nTraining shape:", X_train.shape)
print("Testing shape :", X_test.shape)

print("\nTraining distribution:")
print(y_train.value_counts())

print("\nTesting distribution:")
print(y_test.value_counts())