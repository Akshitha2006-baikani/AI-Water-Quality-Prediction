import pandas as pd
import joblib
import matplotlib.pyplot as plt


# Load dataset
df = pd.read_csv("data/water_quality_prediction.csv")

# Load trained Random Forest
model = joblib.load(
    "models/random_forest_water_quality.pkl"
)

# Features used by the model
X = df.drop(columns=["target_ph_next_day"])

# Get feature importance
importance = model.feature_importances_

# Create DataFrame
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

# Sort from highest to lowest
feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("========== FEATURE IMPORTANCE ==========")
print(feature_importance.to_string(index=False))


# Save results
feature_importance.to_csv(
    "outputs/results/feature_importance.csv",
    index=False
)


# Plot
plt.figure(figsize=(10, 7))

plt.barh(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Water Quality Feature")
plt.title("Feature Importance for Next-Day pH Prediction")

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig(
    "outputs/graphs/feature_importance.png",
    dpi=300
)

plt.show()

print("\nFeature importance results saved successfully.")