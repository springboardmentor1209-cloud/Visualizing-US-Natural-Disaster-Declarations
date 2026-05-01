import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("us_disaster_declarations.csv")

# Display first 5 rows
print(df.head())
print("Dataset Shape:", df.shape)
print(df.columns)
print(df.info())
print(df.describe())
print(df.isnull().sum())
# Convert date columns to datetime
df['declaration_date'] = pd.to_datetime(df['declaration_date'], errors='coerce')
df['incident_begin_date'] = pd.to_datetime(df['incident_begin_date'], errors='coerce')
df['incident_end_date'] = pd.to_datetime(df['incident_end_date'], errors='coerce')
df['disaster_closeout_date'] = pd.to_datetime(df['disaster_closeout_date'], errors='coerce')
df['incident_type'] = df['incident_type'].str.strip().str.title()
df['state'] = df['state'].str.strip().str.upper()
# Forward fill incident end date
df['incident_end_date'] = df['incident_end_date'].ffill()

# Keep datetime column as datetime 
df['disaster_closeout_date'] = df['disaster_closeout_date'].fillna(pd.NaT)

# Fill categorical column safely
df['designated_incident_types'] = df['designated_incident_types'].fillna("Not Specified")
print(df.info())
print(df.isnull().sum())
df['Year'] = df['declaration_date'].dt.year
declarations_per_year = df['Year'].value_counts().sort_index()


# Create Year column from declaration date
df['Year'] = df['declaration_date'].dt.year

# Count declarations per year
declarations_per_year = df['Year'].value_counts().sort_index()

print("\nDeclarations per Year:")
print(declarations_per_year)

# Time Series Plot 
plt.figure(figsize=(10,5))
declarations_per_year.plot()

plt.title("Total Disaster Declarations Per Year")
plt.xlabel("Year")
plt.ylabel("Number of Declarations")

plt.show()


# Count declarations per state
declarations_per_state = df['state'].value_counts()

print("\nTop States:")
print(declarations_per_state.head(10))

# Bar Chart 
plt.figure(figsize=(12,6))
declarations_per_state.head(10).plot(kind='bar')

plt.title("Top 10 States by Disaster Declarations")
plt.xlabel("State")
plt.ylabel("Number of Declarations")
plt.show(block=True)

df.to_csv("cleaned_fema_dataset.csv", index=False)

print("\nCleaned dataset saved successfully!")