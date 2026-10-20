import pandas as pd

X = pd.read_csv("PQC_X_features.csv")
y = pd.read_csv("PQC_y_target.csv")["PQC_label"]

df = X.copy()
df["PQC_label"] = y

print("=" * 80)
print("FEATURE DISTRIBUTION BY PQC CLASS")
print("=" * 80)

for col in X.columns:

    print(f"\n{'-' * 80}")
    print(col)

    stats = df.groupby("PQC_label")[col].agg(
        ["mean", "std", "min", "max"]
    )

    print(stats.round(4))