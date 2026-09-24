import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset

df = pd.read_csv("data/water_quality_prediction.csv")

# Display basic information
print("Dataset Shape:", df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nColumn Names:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDataset Information:")
print(df.info())

print("\nTarget Variable Statistics:")
print(df["target_ph_next_day"].describe())

print("\nNumber of Locations:")
print(df["location_id"].nunique())


# -----------------------------
# Target Variable Distribution
# -----------------------------

plt.figure(figsize=(8, 5))

sns.histplot(
    df["target_ph_next_day"],
    bins=30,
    kde=True
)

plt.title("Distribution of Next-Day pH")
plt.xlabel("Next-Day pH")
plt.ylabel("Frequency")

plt.tight_layout()


plt.savefig("outputs/graphs/target_ph_distribution.png")

plt.show()


# -----------------------------
# Correlation Analysis
# -----------------------------

numeric_df = df.select_dtypes(include="number")

plt.figure(figsize=(12, 8))

sns.heatmap(
    numeric_df.corr(),
    annot=False,
    cmap="coolwarm"
)

plt.title("Correlation Matrix of Water Quality Features")

plt.tight_layout()


plt.savefig("outputs/graphs/correlation_matrix.png")

plt.show()