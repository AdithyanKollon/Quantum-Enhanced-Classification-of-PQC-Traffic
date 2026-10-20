
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif

# Load training data only
X = pd.read_csv("PQC_X_train.csv")
y = pd.read_csv("PQC_y_train.csv").squeeze()

# Calculate mutual information scores
mi_scores = mutual_info_classif(
    X,
    y,
    discrete_features=False,
    n_neighbors=5,
    random_state=42
)

# Rank features from highest to lowest score
ranking = pd.DataFrame({
    "Feature": X.columns,
    "MI_Score": mi_scores
}).sort_values(
    by="MI_Score",
    ascending=False
).reset_index(drop=True)

ranking.index = ranking.index + 1
ranking.index.name = "Rank"

print("\nMUTUAL INFORMATION FEATURE RANKING")
print(ranking.to_string())

# Save ranking
ranking.to_csv("PQC_MI_feature_ranking.csv")

# Plot ranking
plt.figure(figsize=(10, 6))
plt.barh(ranking["Feature"], ranking["MI_Score"])
plt.gca().invert_yaxis()
plt.xlabel("Mutual Information Score")
plt.ylabel("Feature")
plt.title("Mutual Information Feature Ranking — PQC Detection")
plt.tight_layout()
plt.savefig("PQC_MI_feature_ranking.png", dpi=300)
plt.show()
