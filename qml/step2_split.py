import pandas as pd
from sklearn.model_selection import train_test_split

# Load prepared data
X = pd.read_csv("PQC_X_features.csv")
y = pd.read_csv("PQC_y_target.csv")["PQC_label"]

# --------------------------------------------------
# Train-test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# --------------------------------------------------
# Check sizes
# --------------------------------------------------

print("Training set:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nTesting set:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

# --------------------------------------------------
# Check class distribution
# --------------------------------------------------

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nTraining percentages:")
print(
    y_train.value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\nTesting class distribution:")
print(y_test.value_counts())

print("\nTesting percentages:")
print(
    y_test.value_counts(normalize=True)
    .mul(100)
    .round(2)
)

# --------------------------------------------------
# Save splits
# --------------------------------------------------

X_train.to_csv("PQC_X_train.csv", index=False)
X_test.to_csv("PQC_X_test.csv", index=False)

y_train.to_csv(
    "PQC_y_train.csv",
    index=False,
    header=["PQC_label"]
)

y_test.to_csv(
    "PQC_y_test.csv",
    index=False,
    header=["PQC_label"]
)

print("\nCreated:")
print("PQC_X_train.csv")
print("PQC_X_test.csv")
print("PQC_y_train.csv")
print("PQC_y_test.csv")