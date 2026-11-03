import pandas as pd
from ucimlrepo import fetch_ucirepo

print("Downloading Dry Bean Dataset...")

# Fetch dataset
dry_bean = fetch_ucirepo(id=602)

# Extract features and target
X = dry_bean.data.features
y = dry_bean.data.targets

# Combine features and target
df = pd.concat([X, y], axis=1)

# Save dataset
df.to_csv("dataset/dry_bean_dataset.csv", index=False)

print("\nDataset downloaded successfully!")

print("\nDataset Information:")
print("Shape:", df.shape)

print("\nFirst Five Records:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum().sum())

print("\nClass Distribution:")
print(df["Class"].value_counts())