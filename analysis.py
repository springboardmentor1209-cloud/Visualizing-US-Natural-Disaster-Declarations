import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("database.csv")


df.columns = df.columns.str.strip()
df.columns = df.columns.str.replace(" ", "_")
df.columns = df.columns.str.lower()

print("Columns:\n", df.columns)


df['state'] = df['state'].fillna("Unknown")
df['disaster_type'] = df['disaster_type'].fillna("Unknown")


date_cols = ['declaration_date', 'start_date', 'end_date']

for col in date_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')


df = df.drop_duplicates()


df['year'] = df['declaration_date'].dt.year
df['month'] = df['declaration_date'].dt.month


print("\nDeclarations per year:\n")
print(df['year'].value_counts())

print("\nTop states:\n")
print(df['state'].value_counts().head(10))


df['year'].value_counts().sort_index().plot()

plt.title("Disaster Declarations Per Year")
plt.xlabel("Year")
plt.ylabel("Count")
plt.savefig("DDPY.png")
plt.show()


df.to_csv("cleaned_data.csv", index=False)
print(df.columns)
# visualizations
 #Improved year trend
year_counts = df['year'].value_counts().sort_index()

plt.figure()
year_counts.plot(marker='o')

plt.title("Total Disaster Declarations Per Year")
plt.xlabel("Year")
plt.ylabel("Number of Declarations")

plt.grid()
plt.savefig("year_trend.png")
plt.show()
#Disaster types over time
top_types = df['disaster_type'].value_counts().head(5).index
filtered_df = df[df['disaster_type'].isin(top_types)]

disaster_trend = filtered_df.groupby(['year', 'disaster_type']).size().unstack(fill_value=0)

plt.figure()
disaster_trend.plot()

plt.title("Top 5 Disaster Types Over Time")
plt.xlabel("Year")
plt.ylabel("Number of Disasters")

plt.legend(title="Disaster Type")
plt.grid()
plt.savefig("Top5.png")
plt.show()
# Seasonality
df['month'] = df['declaration_date'].dt.month

month_counts = df['month'].value_counts().sort_index()

plt.figure()
month_counts.plot(kind='bar')

plt.title("Disaster Occurrence by Month")
plt.xlabel("Month")
plt.ylabel("Count")
plt.savefig("Occurence.png")
plt.show()
#Average Disaster Duration

df = df.dropna(subset=['start_date', 'end_date'])


df['duration'] = (df['end_date'] - df['start_date']).dt.days

duration_trend = df.groupby('year')['duration'].mean()

disaster_trend.plot()

plt.title("Average Disaster Duration Over Time")
plt.xlabel("Year")
plt.ylabel("Duration (Days)")

plt.grid()
plt.savefig("over time.png")
plt.show()

