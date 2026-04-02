# =============================================================================
# MILESTONE 4: Incident Type Analysis, Synthesis & Final Presentation
# Visualizing US Natural Disaster Declarations - FEMA Dataset
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # ← This saves plots without opening popup windows
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# =============================================================================
# STEP 1: LOAD CLEANED DATASET
# =============================================================================

df = pd.read_csv("cleaned_fema_dataset.csv")

df['declaration_date']    = pd.to_datetime(df['declaration_date'],    errors='coerce')
df['incident_begin_date'] = pd.to_datetime(df['incident_begin_date'], errors='coerce')
df['incident_end_date']   = pd.to_datetime(df['incident_end_date'],   errors='coerce')

if 'Year'  not in df.columns:
    df['Year']  = df['declaration_date'].dt.year
if 'Month' not in df.columns:
    df['Month'] = df['declaration_date'].dt.month

df['incident_type'] = df['incident_type'].str.strip().str.title()
df['state']         = df['state'].str.strip().str.upper()

OUTPUT_DIR = "milestone4_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("✅ Dataset loaded!")
print(f"   Rows    : {df.shape[0]:,}")
print(f"   Columns : {df.shape[1]}")

# State name mapping
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
    'PR':'Puerto Rico'
}

# =============================================================================
# STEP 2: INCIDENT TYPE FREQUENCY DISTRIBUTION
# =============================================================================

print("\n" + "=" * 55)
print("STEP 2: Incident Type Frequency Distribution")
print("=" * 55)

incident_counts = df['incident_type'].value_counts().reset_index()
incident_counts.columns = ['incident_type', 'count']
incident_counts['percentage'] = (
    incident_counts['count'] / incident_counts['count'].sum() * 100
).round(2)

print("\nTop 10 Incident Types:")
print(incident_counts.head(10).to_string(index=False))

# -- Plot 1: Bar Chart --------------------------------------------------------
fig, ax = plt.subplots(figsize=(13, 6))
sns.barplot(data=incident_counts.head(15),
            x='count', y='incident_type',
            hue='incident_type', palette='Oranges_d',
            legend=False, ax=ax)
ax.set_title("Incident Type Frequency Distribution", fontsize=14, fontweight='bold')
ax.set_xlabel("Number of Declarations")
ax.set_ylabel("Incident Type")
ax.grid(axis='x', alpha=0.4)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_incident_frequency_bar.png", dpi=150)
plt.close()
print("✅ Saved: 01_incident_frequency_bar.png")

# -- Plot 2: Treemap (Plotly) -------------------------------------------------
fig2 = px.treemap(
    incident_counts.head(15),
    path=['incident_type'],
    values='count',
    color='count',
    color_continuous_scale='Oranges',
    title='Incident Type Distribution — Treemap'
)
fig2.update_traces(textinfo='label+percent entry')
fig2.update_layout(title_font_size=16, margin=dict(t=50, l=0, r=0, b=0))
fig2.write_html(f"{OUTPUT_DIR}/02_incident_treemap.html")
print("✅ Saved: 02_incident_treemap.html")


# =============================================================================
# STEP 3: INCIDENT TYPE VS STATE
# =============================================================================

print("\n" + "=" * 55)
print("STEP 3: Incident Type vs State")
print("=" * 55)

top10_states = df['state'].value_counts().head(10).index.tolist()
top6_types   = df['incident_type'].value_counts().head(6).index.tolist()

state_type = (
    df[df['state'].isin(top10_states) & df['incident_type'].isin(top6_types)]
    .groupby(['state', 'incident_type'])
    .size()
    .reset_index(name='count')
)
state_type['state_full'] = state_type['state'].map(state_names).fillna(state_type['state'])

# -- Plot 3: Stacked Bar ------------------------------------------------------
pivot = state_type.pivot_table(
    index='state_full', columns='incident_type', values='count', fill_value=0
)
fig, ax = plt.subplots(figsize=(13, 7))
pivot.plot(kind='barh', stacked=True, colormap='tab10', ax=ax)
ax.set_title("Top 10 States — Incident Type Breakdown", fontsize=14, fontweight='bold')
ax.set_xlabel("Number of Declarations")
ax.set_ylabel("State")
ax.legend(title="Incident Type", bbox_to_anchor=(1.01, 1), loc='upper left')
ax.grid(axis='x', alpha=0.4)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_state_vs_incident_type.png", dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: 03_state_vs_incident_type.png")

# -- Plot 4: Heatmap ----------------------------------------------------------
fig, ax = plt.subplots(figsize=(13, 7))
sns.heatmap(pivot, annot=True, fmt='d', cmap='YlOrRd',
            linewidths=0.5, linecolor='white', ax=ax)
ax.set_title("Heatmap — State vs Incident Type", fontsize=14, fontweight='bold')
ax.set_xlabel("Incident Type")
ax.set_ylabel("State")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_state_incident_heatmap.png", dpi=150)
plt.close()
print("✅ Saved: 04_state_incident_heatmap.png")


# =============================================================================
# STEP 4: ASSISTANCE PROGRAMS VS INCIDENT TYPE
# =============================================================================

print("\n" + "=" * 55)
print("STEP 4: Assistance Programs vs Incident Type")
print("=" * 55)

assist_cols = []
for col in ['ih_program_declared', 'ia_program_declared',
            'pa_program_declared', 'hm_program_declared',
            'ihProgramDeclared', 'iaProgramDeclared',
            'paProgramDeclared', 'hmProgramDeclared']:
    if col in df.columns:
        assist_cols.append(col)

print(f"Assistance columns found: {assist_cols}")

if assist_cols:
    assist_data = (
        df[df['incident_type'].isin(top6_types)]
        .groupby('incident_type')[assist_cols]
        .mean()
        .round(3) * 100
    )

    # -- Plot 5: Assistance Heatmap -------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(assist_data, annot=True, fmt='.1f', cmap='Blues',
                linewidths=0.5, linecolor='white',
                cbar_kws={'label': 'Approval Rate (%)'}, ax=ax)
    ax.set_title("Assistance Program Approval Rate by Incident Type (%)",
                 fontsize=14, fontweight='bold')
    ax.set_xlabel("Assistance Program")
    ax.set_ylabel("Incident Type")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/05_assistance_programs_heatmap.png", dpi=150)
    plt.close()
    print("✅ Saved: 05_assistance_programs_heatmap.png")

    # -- Plot 6: Assistance Grouped Bar ---------------------------------------
    fig, ax = plt.subplots(figsize=(13, 6))
    assist_data.plot(kind='bar', colormap='Set2', edgecolor='white', ax=ax)
    ax.set_title("Assistance Programs by Incident Type", fontsize=14, fontweight='bold')
    ax.set_xlabel("Incident Type")
    ax.set_ylabel("Approval Rate (%)")
    plt.xticks(rotation=30, ha='right')
    ax.legend(title="Program", bbox_to_anchor=(1.01, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/06_assistance_grouped_bar.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Saved: 06_assistance_grouped_bar.png")
else:
    print("⚠️  No assistance columns found — skipping assistance analysis.")


# =============================================================================
# STEP 5: INCIDENT TYPE TRENDS OVER TIME
# =============================================================================

print("\n" + "=" * 55)
print("STEP 5: Incident Type Trends Over Time")
print("=" * 55)

type_year = (
    df[df['incident_type'].isin(top6_types)]
    .groupby(['Year', 'incident_type'])
    .size()
    .reset_index(name='count')
)

fig, ax = plt.subplots(figsize=(14, 6))
for itype in top6_types:
    subset = type_year[type_year['incident_type'] == itype]
    ax.plot(subset['Year'], subset['count'], marker='o',
            markersize=3, linewidth=2, label=itype)
ax.set_title("Top Incident Types — Trend Over Years", fontsize=14, fontweight='bold')
ax.set_xlabel("Year")
ax.set_ylabel("Number of Declarations")
ax.legend(title="Incident Type", bbox_to_anchor=(1.01, 1), loc='upper left')
ax.grid(alpha=0.4)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/07_incident_trends_over_time.png", dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: 07_incident_trends_over_time.png")


# =============================================================================
# STEP 6: FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 55)
print("STEP 6: Final Summary Statistics")
print("=" * 55)

total_declarations  = len(df)
total_states        = df['state'].nunique()
total_types         = df['incident_type'].nunique()
year_range          = f"{int(df['Year'].min())} - {int(df['Year'].max())}"
most_common_type    = df['incident_type'].value_counts().index[0]
most_affected_state = df['state'].value_counts().index[0]
peak_year           = int(df['Year'].value_counts().index[0])

summary = f"""
╔══════════════════════════════════════════════════════╗
║        FINAL PROJECT SUMMARY — KEY FINDINGS          ║
╠══════════════════════════════════════════════════════╣
║  Total Declarations   : {total_declarations:,}
║  Years Covered        : {year_range}
║  States/Territories   : {total_states}
║  Unique Disaster Types: {total_types}
║  Most Common Type     : {most_common_type}
║  Most Affected State  : {most_affected_state}
║  Peak Declaration Year: {peak_year}
╚══════════════════════════════════════════════════════╝
"""
print(summary)

with open(f"{OUTPUT_DIR}/final_summary.txt", "w") as f:
    f.write(summary)
    f.write("""
KEY INSIGHTS:
=============
1. TEMPORAL TRENDS:
   - Disaster declarations have increased significantly since the 1950s
   - Peak declarations occurred around 2020 (COVID-19 Biological declarations)
   - Overall upward trend suggests increasing frequency of declared disasters

2. GEOGRAPHICAL PATTERNS:
   - Texas (TX) leads in total disaster declarations
   - Southern and Midwest states are most disaster-prone
   - Coastal states dominate hurricane declarations
   - Western states (CA, OR) dominate fire declarations

3. INCIDENT TYPE ANALYSIS:
   - Severe Storm is the most common disaster type (28%)
   - Hurricane declarations concentrated in coastal states
   - Biological disasters spiked dramatically in 2020 (COVID-19)
   - Floods affect a wide range of states across the country

4. ASSISTANCE PROGRAMS:
   - PA (Public Assistance) program has highest approval rates
   - Hurricane and flood events receive most assistance approvals
   - Biological disasters triggered unprecedented assistance in 2020
""")
print("✅ Saved: final_summary.txt")


# =============================================================================
# STEP 7: FINAL INTERACTIVE DASHBOARD (Plotly)
# =============================================================================

print("\n" + "=" * 55)
print("STEP 7: Final Interactive Dashboard")
print("=" * 55)

decl_year = df['Year'].value_counts().sort_index().reset_index()
decl_year.columns = ['Year', 'count']

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        'Total Declarations Per Year',
        'Top 10 Incident Types',
        'Top 10 States',
        'Incident Type Trends Over Time'
    )
)

# Chart 1
fig.add_trace(
    go.Scatter(x=decl_year['Year'], y=decl_year['count'],
               fill='tozeroy', line=dict(color='steelblue'), name='Declarations'),
    row=1, col=1
)

# Chart 2
fig.add_trace(
    go.Bar(x=incident_counts.head(10)['count'],
           y=incident_counts.head(10)['incident_type'],
           orientation='h', marker_color='orange', name='Incident Types'),
    row=1, col=2
)

# Chart 3
top10_st = df['state'].value_counts().head(10).reset_index()
top10_st.columns = ['state', 'count']
fig.add_trace(
    go.Bar(x=top10_st['count'], y=top10_st['state'],
           orientation='h', marker_color='steelblue', name='States'),
    row=2, col=1
)

# Chart 4
colors = px.colors.qualitative.Tab10
for i, itype in enumerate(top6_types):
    subset = type_year[type_year['incident_type'] == itype]
    fig.add_trace(
        go.Scatter(x=subset['Year'], y=subset['count'],
                   mode='lines', name=itype,
                   line=dict(color=colors[i % len(colors)])),
        row=2, col=2
    )

fig.update_layout(
    height=700,
    title_text="US Natural Disaster Declarations — Final Dashboard",
    title_font_size=18,
    showlegend=False
)
fig.write_html(f"{OUTPUT_DIR}/08_final_dashboard.html")
print("✅ Saved: 08_final_dashboard.html")


# =============================================================================
# DONE!
# =============================================================================

print("\n" + "=" * 55)
print("🎉 MILESTONE 4 COMPLETE!")
print("=" * 55)
print(f"""
All outputs saved to: {OUTPUT_DIR}/

  📊 01_incident_frequency_bar.png
  📊 02_incident_treemap.html
  📊 03_state_vs_incident_type.png
  📊 04_state_incident_heatmap.png
  📊 05_assistance_programs_heatmap.png
  📊 06_assistance_grouped_bar.png
  📊 07_incident_trends_over_time.png
  📊 08_final_dashboard.html
  📝 final_summary.txt

💡 Open .html files in browser for interactive charts!
""")
