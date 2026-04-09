# =============================================================================
# MILESTONE 3: Geographical Pattern Visualization
# Visualizing US Natural Disaster Declarations - FEMA Dataset
# =============================================================================

# ── INSTALL (run once in terminal if needed) ──────────────────────────────────
# pip install pandas plotly folium geopandas

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import folium
from folium.plugins import HeatMap
import os

# =============================================================================
# STEP 1: LOAD CLEANED DATASET (from Milestone 1)
# =============================================================================

df = pd.read_csv("cleaned_fema_dataset.csv")

# Convert dates if not already
df['declaration_date'] = pd.to_datetime(df['declaration_date'], errors='coerce')

# Make sure Year column exists
if 'Year' not in df.columns:
    df['Year'] = df['declaration_date'].dt.year

# Standardize columns
df['incident_type'] = df['incident_type'].str.strip().str.title()
df['state']         = df['state'].str.strip().str.upper()

print("✅ Dataset loaded!")
print(f"   Rows    : {df.shape[0]:,}")
print(f"   Columns : {df.shape[1]}")
print(f"\nStates in dataset: {df['state'].nunique()}")
print(f"Incident types   : {df['incident_type'].nunique()}")

# State abbreviation → Full name mapping (for labels)
state_names = {
    'AL':'Alabama','AK':'Alaska','AZ':'Arizona','AR':'Arkansas','CA':'California',
    'CO':'Colorado','CT':'Connecticut','DE':'Delaware','FL':'Florida','GA':'Georgia',
    'HI':'Hawaii','ID':'Idaho','IL':'Illinois','IN':'Indiana','IA':'Iowa',
    'KS':'Kansas','KY':'Kentucky','LA':'Louisiana','ME':'Maine','MD':'Maryland',
    'MA':'Massachusetts','MI':'Michigan','MN':'Minnesota','MS':'Mississippi',
    'MO':'Missouri','MT':'Montana','NE':'Nebraska','NV':'Nevada','NH':'New Hampshire',
    'NJ':'New Jersey','NM':'New Mexico','NY':'New York','NC':'North Carolina',
    'ND':'North Dakota','OH':'Ohio','OK':'Oklahoma','OR':'Oregon','PA':'Pennsylvania',
    'RI':'Rhode Island','SC':'South Carolina','SD':'South Dakota','TN':'Tennessee',
    'TX':'Texas','UT':'Utah','VT':'Vermont','VA':'Virginia','WA':'Washington',
    'WV':'West Virginia','WI':'Wisconsin','WY':'Wyoming','DC':'District of Columbia',
    'PR':'Puerto Rico','GU':'Guam','VI':'Virgin Islands','AS':'American Samoa',
    'MP':'Northern Mariana Islands'
}
df['state_full'] = df['state'].map(state_names).fillna(df['state'])

# Output folder
OUTPUT_DIR = "milestone3_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# STEP 2: AGGREGATE DATA BY STATE
# =============================================================================

# Total declarations per state
state_counts = df['state'].value_counts().reset_index()
state_counts.columns = ['state', 'total_declarations']
state_counts['state_full'] = state_counts['state'].map(state_names).fillna(state_counts['state'])

print("\nTop 10 States by Total Declarations:")
print(state_counts.head(10).to_string(index=False))

# Declarations per state per incident type
state_incident = (
    df.groupby(['state', 'incident_type'])
    .size()
    .reset_index(name='count')
)

# Top 5 disaster types for hotspot analysis
top5_types = df['incident_type'].value_counts().head(5).index.tolist()
print(f"\nTop 5 Incident Types: {top5_types}")


# =============================================================================
# STEP 3: CHOROPLETH MAP — Total Declarations Per State (Plotly)
# =============================================================================

print("\n" + "=" * 55)
print("STEP 3: Choropleth Map - Total Declarations Per State")
print("=" * 55)

fig = px.choropleth(
    state_counts,
    locations='state',
    locationmode='USA-states',
    color='total_declarations',
    scope='usa',
    color_continuous_scale='Blues',
    hover_name='state_full',
    hover_data={'total_declarations': True, 'state': False},
    title='Total FEMA Disaster Declarations by State',
    labels={'total_declarations': 'Total Declarations'}
)
fig.update_layout(
    title_font_size=18,
    geo=dict(showlakes=True, lakecolor='lightblue'),
    coloraxis_colorbar=dict(title="Declarations"),
    margin=dict(l=0, r=0, t=50, b=0)
)
fig.write_html(f"{OUTPUT_DIR}/01_choropleth_total_declarations.html")
fig.show()
print("✅ Saved: 01_choropleth_total_declarations.html")


# =============================================================================
# STEP 4: CHOROPLETH MAPS — Hotspots for Top Disaster Types (Plotly)
# =============================================================================

print("\n" + "=" * 55)
print("STEP 4: Hotspot Choropleth Maps by Disaster Type")
print("=" * 55)

hotspot_types = ['Hurricane', 'Flood', 'Fire', 'Severe Storm', 'Tornado']

for i, disaster in enumerate(hotspot_types, start=1):
    subset = state_incident[state_incident['incident_type'] == disaster].copy()
    if subset.empty:
        print(f"⚠️  No data for: {disaster}")
        continue

    subset['state_full'] = subset['state'].map(state_names).fillna(subset['state'])

    fig = px.choropleth(
        subset,
        locations='state',
        locationmode='USA-states',
        color='count',
        scope='usa',
        color_continuous_scale='Reds',
        hover_name='state_full',
        hover_data={'count': True, 'state': False},
        title=f'FEMA Disaster Hotspots — {disaster}',
        labels={'count': 'Declarations'}
    )
    fig.update_layout(
        title_font_size=16,
        geo=dict(showlakes=True, lakecolor='lightblue'),
        margin=dict(l=0, r=0, t=50, b=0)
    )
    fname = f"{OUTPUT_DIR}/0{i+1}_hotspot_{disaster.lower().replace(' ', '_')}.html"
    fig.write_html(fname)
    fig.show()
    print(f"✅ Saved: {fname}")


# =============================================================================
# STEP 5: FOLIUM HEATMAP — Disaster Intensity Across States
# =============================================================================

print("\n" + "=" * 55)
print("STEP 5: Folium Heatmap - Disaster Intensity")
print("=" * 55)

# State center coordinates for heatmap
state_coords = {
    'AL':(32.8,-86.8),'AK':(64.2,-153.4),'AZ':(34.3,-111.1),'AR':(34.9,-92.4),
    'CA':(36.8,-119.4),'CO':(39.1,-105.4),'CT':(41.6,-72.7),'DE':(38.9,-75.5),
    'FL':(28.7,-82.5),'GA':(32.2,-83.4),'HI':(20.3,-156.4),'ID':(44.4,-114.6),
    'IL':(40.0,-89.2),'IN':(39.9,-86.3),'IA':(42.1,-93.5),'KS':(38.5,-98.4),
    'KY':(37.7,-84.9),'LA':(31.1,-91.9),'ME':(45.3,-69.2),'MD':(39.1,-76.8),
    'MA':(42.4,-71.4),'MI':(44.3,-85.4),'MN':(46.4,-93.1),'MS':(32.7,-89.7),
    'MO':(38.5,-92.5),'MT':(47.0,-109.6),'NE':(41.5,-99.9),'NV':(39.3,-116.6),
    'NH':(43.7,-71.6),'NJ':(40.2,-74.7),'NM':(34.4,-106.1),'NY':(42.9,-75.5),
    'NC':(35.6,-79.8),'ND':(47.5,-100.5),'OH':(40.4,-82.8),'OK':(35.6,-96.9),
    'OR':(44.1,-120.5),'PA':(40.6,-77.2),'RI':(41.7,-71.5),'SC':(33.9,-80.9),
    'SD':(44.4,-100.2),'TN':(35.9,-86.7),'TX':(31.5,-99.3),'UT':(39.4,-111.1),
    'VT':(44.0,-72.7),'VA':(37.5,-78.5),'WA':(47.4,-120.4),'WV':(38.9,-80.5),
    'WI':(44.3,-89.8),'WY':(43.0,-107.6),'DC':(38.9,-77.0),'PR':(18.2,-66.6)
}

# Build heatmap data: [lat, lon, weight]
heat_data = []
for _, row in state_counts.iterrows():
    if row['state'] in state_coords:
        lat, lon = state_coords[row['state']]
        heat_data.append([lat, lon, row['total_declarations']])

# Create folium map
m = folium.Map(location=[37.8, -96.9], zoom_start=4, tiles='CartoDB positron')
HeatMap(
    heat_data,
    min_opacity=0.3,
    max_zoom=6,
    radius=40,
    blur=30,
    gradient={0.2: 'blue', 0.5: 'lime', 0.8: 'orange', 1.0: 'red'}
).add_to(m)

# Add state markers with declaration count
for _, row in state_counts.iterrows():
    if row['state'] in state_coords:
        lat, lon = state_coords[row['state']]
        folium.CircleMarker(
            location=[lat, lon],
            radius=5,
            color='darkblue',
            fill=True,
            fill_opacity=0.6,
            tooltip=f"{row['state_full']}: {row['total_declarations']:,} declarations"
        ).add_to(m)

folium_path = f"{OUTPUT_DIR}/07_folium_heatmap.html"
m.save(folium_path)
print(f"✅ Saved: {folium_path}")


# =============================================================================
# STEP 6: BAR CHART — Top 20 States (Matplotlib/Seaborn)
# =============================================================================

print("\n" + "=" * 55)
print("STEP 6: Top 20 States Bar Chart")
print("=" * 55)

top20 = state_counts.head(20)

plt.figure(figsize=(13, 7))
sns.set_theme(style="whitegrid")
bars = sns.barplot(data=top20, x='total_declarations', y='state_full', palette='Blues_d')

# Add value labels
for i, (val, name) in enumerate(zip(top20['total_declarations'], top20['state_full'])):
    bars.text(val + 30, i, f'{val:,}', va='center', fontsize=9)

plt.title("Top 20 States by Total FEMA Disaster Declarations", fontsize=14, fontweight='bold')
plt.xlabel("Total Declarations")
plt.ylabel("State")
plt.grid(axis='x', alpha=0.4)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/08_top20_states_bar.png", dpi=150)
plt.show()
print("✅ Saved: 08_top20_states_bar.png")


# =============================================================================
# STEP 7: STACKED BAR — Top 10 States × Top 5 Disaster Types
# =============================================================================

print("\n" + "=" * 55)
print("STEP 7: Top States × Disaster Types Stacked Bar")
print("=" * 55)

top10_states = state_counts.head(10)['state'].tolist()

state_type_pivot = (
    state_incident[
        (state_incident['state'].isin(top10_states)) &
        (state_incident['incident_type'].isin(top5_types))
    ]
    .pivot_table(index='state', columns='incident_type', values='count', fill_value=0)
)

# Add full state names
state_type_pivot.index = [state_names.get(s, s) for s in state_type_pivot.index]

state_type_pivot.plot(
    kind='barh',
    stacked=True,
    figsize=(13, 7),
    colormap='tab10'
)
plt.title("Top 10 States — Breakdown by Top 5 Disaster Types", fontsize=14, fontweight='bold')
plt.xlabel("Number of Declarations")
plt.ylabel("State")
plt.legend(title="Disaster Type", bbox_to_anchor=(1.01, 1), loc='upper left')
plt.grid(axis='x', alpha=0.4)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/09_states_by_type_stacked.png", dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: 09_states_by_type_stacked.png")


# =============================================================================
# STEP 8: SUMMARY
# =============================================================================

print("\n" + "=" * 55)
print("🎉 MILESTONE 3 COMPLETE!")
print("=" * 55)
print(f"\nAll outputs saved to: {OUTPUT_DIR}/")
print("""
Files generated:
  📊 01_choropleth_total_declarations.html  → Open in browser
  📊 02_hotspot_hurricane.html              → Open in browser
  📊 03_hotspot_flood.html                  → Open in browser
  📊 04_hotspot_fire.html                   → Open in browser
  📊 05_hotspot_severe_storm.html           → Open in browser
  📊 06_hotspot_tornado.html                → Open in browser
  📊 07_folium_heatmap.html                 → Open in browser
  📊 08_top20_states_bar.png                → Image
  📊 09_states_by_type_stacked.png          → Image

💡 Tip: Open all .html files in your browser for
         interactive maps with hover tooltips!
""")
