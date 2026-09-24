import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------------------------
# 1. Load Dataset
# --------------------------------------------------

df = pd.read_csv("data/water_quality_prediction.csv")

print("Dataset loaded successfully.")
print("Shape:", df.shape)


# --------------------------------------------------
# 2. Define Features and Target
# --------------------------------------------------

X = df.drop(columns=["target_ph_next_day"])
y = df["target_ph_next_day"]

print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget: target_ph_next_day")


# --------------------------------------------------
# 3. Train-Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# --------------------------------------------------
# 4. Linear Regression
# --------------------------------------------------

linear_model = LinearRegression()

linear_model.fit(X_train, y_train)

linear_predictions = linear_model.predict(X_test)

linear_mae = mean_absolute_error(
    y_test,
    linear_predictions
)

linear_rmse = mean_squared_error(
    y_test,
    linear_predictions
) ** 0.5

linear_r2 = r2_score(
    y_test,
    linear_predictions
)

print("\n========== Linear Regression ==========")
print("MAE :", linear_mae)
print("RMSE:", linear_rmse)
print("R²  :", linear_r2)


# --------------------------------------------------
# 5. Random Forest
# --------------------------------------------------

random_forest = RandomForestRegressor(
    n_estimators=30,
    random_state=42,
    n_jobs=-1
)

random_forest.fit(X_train, y_train)

rf_predictions = random_forest.predict(X_test)

rf_mae = mean_absolute_error(
    y_test,
    rf_predictions
)

rf_rmse = mean_squared_error(
    y_test,
    rf_predictions
) ** 0.5

rf_r2 = r2_score(
    y_test,
    rf_predictions
)

print("\n========== Random Forest ==========")
print("MAE :", rf_mae)
print("RMSE:", rf_rmse)
print("R²  :", rf_r2)


# --------------------------------------------------
# 6. Compare Models
# --------------------------------------------------

print("\n========== MODEL COMPARISON ==========")

print(
    f"Linear Regression -> "
    f"MAE: {linear_mae:.6f}, "
    f"RMSE: {linear_rmse:.6f}, "
    f"R²: {linear_r2:.6f}"
)

print(
    f"Random Forest -> "
    f"MAE: {rf_mae:.6f}, "
    f"RMSE: {rf_rmse:.6f}, "
    f"R²: {rf_r2:.6f}"
)


# --------------------------------------------------
# 7. Save Random Forest Model
# --------------------------------------------------

joblib.dump(
    random_forest,
    "models/random_forest_water_quality.pkl"
)

print("\nRandom Forest model saved successfully.")