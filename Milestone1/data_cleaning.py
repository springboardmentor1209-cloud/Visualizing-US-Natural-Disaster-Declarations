import pandas as pd

# Load dataset
df = pd.read_csv("../data/database.csv", on_bad_lines="skip", engine="python")

# Show first 5 rows
print("First 5 rows:")
print(df.head())

# Show column names
print("\nColumn names:")
print(df.columns)

# Show dataset info
print("\nDataset Info:")
print(df.info())

# Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# Remove duplicate rows
df = df.drop_duplicates()

# Fill missing values forward
df = df.ffill()

# Convert date columns if present
for col in df.columns:
    if "date" in col.lower():
        df[col] = pd.to_datetime(df[col], errors="coerce")

# Save cleaned dataset
df.to_csv("../data/cleaned_data.csv", index=False)

print("\nData cleaning completed successfully!")
print("Cleaned file saved as cleaned_data.csv")