import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load dataset
df = pd.read_csv("data/water_quality_prediction.csv")

# Load trained model
model = joblib.load("models/random_forest_water_quality.pkl")

# Separate features and target
X = df.drop(columns=["target_ph_next_day"])
y = df["target_ph_next_day"]

# Same train-test split used during training
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Generate predictions
predictions = model.predict(X_test)

# Calculate metrics
mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

print("========== FINAL MODEL RESULTS ==========")
print(f"MAE : {mae:.6f}")
print(f"RMSE: {rmse:.6f}")
print(f"R²  : {r2:.6f}")


# ------------------------------------------------
# Actual vs Predicted
# ------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    predictions,
    alpha=0.5
)

plt.xlabel("Actual Next-Day pH")
plt.ylabel("Predicted Next-Day pH")
plt.title("Actual vs Predicted Next-Day pH")

# Reference line
minimum = min(y_test.min(), predictions.min())
maximum = max(y_test.max(), predictions.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.tight_layout()

plt.savefig(
    "outputs/graphs/actual_vs_predicted.png",
    dpi=300
)

plt.show()


# ------------------------------------------------
# Sample Predictions
# ------------------------------------------------

results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": predictions
})

print("\n========== SAMPLE PREDICTIONS ==========")
print(results.head(20))

# Save prediction results
results.to_csv(
    "outputs/results/predictions.csv",
    index=False
)

print("\nPrediction results saved successfully.")