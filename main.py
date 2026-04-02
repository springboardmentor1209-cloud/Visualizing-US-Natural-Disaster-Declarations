# =============================================================================
# FEMA Disaster Declarations – Data Analysis
# =============================================================================
# Goals:
#   1. Acquire & load the dataset into a Pandas DataFrame.
#   2. Clean the data: fix dtypes, handle missing values, standardize categories.
#   3. EDA: summary statistics, value distributions, frequency counts.
#   4. Plots: time-series of declarations per year, bar chart per state.
# =============================================================================

import pandas as pd
import matplotlib
matplotlib.use("Agg")          # save-to-file; works in any terminal
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Matplotlib style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f0f23",
    "axes.facecolor":   "#1a1a3e",
    "axes.edgecolor":   "#444466",
    "axes.labelcolor":  "#ccccff",
    "xtick.color":      "#aaaacc",
    "ytick.color":      "#aaaacc",
    "text.color":       "#eeeeee",
    "grid.color":       "#2a2a5a",
    "grid.alpha":       0.5,
    "font.family":      "DejaVu Sans",
})

ACCENT    = "#7b61ff"
HIGHLIGHT = "#ff6b6b"

# =============================================================================
# 1. DATA LOADING
# =============================================================================
CSV_PATH = "../data/DisasterDeclarationsSummaries.csv"

print("=" * 60)
print("STEP 1 – Loading data …")
print("=" * 60)

df_raw = pd.read_csv(CSV_PATH, low_memory=False)
print(f"  Raw shape : {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")
print(f"  Columns   : {list(df_raw.columns)}\n")

# =============================================================================
# 2. DATA CLEANING
# =============================================================================
print("=" * 60)
print("STEP 2 – Cleaning data …")
print("=" * 60)

df = df_raw.copy()

# ── 2a. Fix date columns ──────────────────────────────────────────────────────
DATE_COLS = [c for c in df.columns if "date" in c.lower() or "Date" in c]
for col in DATE_COLS:
    df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
print(f"  Parsed date columns : {DATE_COLS}")

# ── 2b. Standardise categorical text ─────────────────────────────────────────
# Dynamically pick whichever variant of each column exists in this CSV
CAT_COL_CANDIDATES = ["incidentType", "declarationType", "state", "stateCode",
                       "programTypeCode", "designatedArea"]
CAT_COLS = [c for c in CAT_COL_CANDIDATES if c in df.columns]
for col in CAT_COLS:
    df[col] = df[col].astype(str).str.strip().str.upper()
print(f"  Standardised cats   : {CAT_COLS}")

# ── 2c. Drop fully-empty rows ─────────────────────────────────────────────────
before = len(df)
df.dropna(how="all", inplace=True)
print(f"  All-NaN rows dropped: {before - len(df)}")

# ── 2d. Missing value report ──────────────────────────────────────────────────
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
print("\n  Columns with missing values (top 10):")
print(missing.head(10).to_string())
print()

# ── 2e. Extract helper columns ────────────────────────────────────────────────
if "declarationDate" in df.columns:
    df["year"]  = df["declarationDate"].dt.year
    df["month"] = df["declarationDate"].dt.month

print(f"  Final shape after cleaning: {df.shape[0]:,} rows × {df.shape[1]} columns\n")

# =============================================================================
# 3. EXPLORATORY DATA ANALYSIS  (EDA)
# =============================================================================
print("=" * 60)
print("STEP 3 – Exploratory Data Analysis")
print("=" * 60)

# ── 3a. Summary statistics for numeric columns ────────────────────────────────
numeric_cols = df.select_dtypes(include="number").columns.tolist()
if numeric_cols:
    print("\n  Numeric summary statistics:")
    print(df[numeric_cols].describe().to_string())

# ── 3b. Incident-type distribution ───────────────────────────────────────────
if "incidentType" in df.columns:
    print("\n  Incident type distribution (top 15):")
    print(df["incidentType"].value_counts().head(15).to_string())

# ── 3c. Declaration-type distribution ────────────────────────────────────────
if "declarationType" in df.columns:
    print("\n  Declaration type distribution:")
    print(df["declarationType"].value_counts().to_string())

# ── 3d. Declarations per year ────────────────────────────────────────────────
if "year" in df.columns:
    decl_per_year = df.groupby("year").size().reset_index(name="count")
    print("\n  Declarations per year (last 10 years):")
    print(decl_per_year.tail(10).to_string(index=False))

# ── 3e. Declarations per state ───────────────────────────────────────────────
STATE_COL = next((c for c in ["state", "stateCode"] if c in df.columns), None)
if STATE_COL:
    decl_per_state = df[STATE_COL].value_counts().reset_index()
    decl_per_state.columns = ["state", "count"]
    print(f"\n  Top 15 states by declaration count:")
    print(decl_per_state.head(15).to_string(index=False))

print()

# =============================================================================
# 4. VISUALISATIONS
# =============================================================================
print("=" * 60)
print("STEP 4 – Generating plots …")
print("=" * 60)

# ── FIGURE 1 : Time-series – total declarations per year ─────────────────────
if "year" in df.columns:
    yearly = df.groupby("year").size()
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(yearly.index, yearly.values, alpha=0.25, color=ACCENT)
    ax.plot(yearly.index, yearly.values, color=ACCENT, linewidth=2.5,
            marker="o", markersize=5)

    # Annotate the peak year
    peak_year = yearly.idxmax()
    peak_val  = yearly.max()
    ax.annotate(
        f"Peak: {peak_year}\n({peak_val:,})",
        xy=(peak_year, peak_val),
        xytext=(peak_year + 2, peak_val * 0.95),
        arrowprops=dict(arrowstyle="->", color=HIGHLIGHT, lw=1.5),
        color=HIGHLIGHT, fontsize=9,
    )

    ax.set_title("FEMA Disaster Declarations — Total Per Year",
                 fontsize=15, fontweight="bold", pad=14)
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Number of Declarations", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.grid(axis="y")
    plt.tight_layout()
    plt.savefig("plot_declarations_per_year.png", dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print("  Saved: plot_declarations_per_year.png")

# ── FIGURE 2 : Bar chart – declarations per state (top 20) ───────────────────
if STATE_COL:
    top_states = (
        df[STATE_COL]
        .value_counts()
        .head(20)
        .sort_values()
    )

    fig, ax = plt.subplots(figsize=(12, 8))
    colors = [HIGHLIGHT if s == top_states.idxmax() else ACCENT
              for s in top_states.index]
    bars = ax.barh(top_states.index, top_states.values,
                   color=colors, edgecolor="#ffffff22", linewidth=0.5)

    # Value labels on bars
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 5, bar.get_y() + bar.get_height() / 2,
                f"{int(w):,}", va="center", ha="left",
                fontsize=8.5, color="#ccccff")

    ax.set_title("FEMA Declarations by State — Top 20",
                 fontsize=15, fontweight="bold", pad=14)
    ax.set_xlabel("Number of Declarations", fontsize=11)
    ax.set_ylabel("State Code", fontsize=11)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.grid(axis="x")
    plt.tight_layout()
    plt.savefig("plot_declarations_per_state.png", dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print("  Saved: plot_declarations_per_state.png")

# ── FIGURE 3 : Bar chart – top incident types ────────────────────────────────
if "incidentType" in df.columns:
    top_types = (
        df["incidentType"]
        .value_counts()
        .head(15)
        .sort_values()
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    bar_colors = [HIGHLIGHT if t == top_types.idxmax() else "#5e5ce6"
                  for t in top_types.index]
    ax.barh(top_types.index, top_types.values,
            color=bar_colors, edgecolor="#ffffff22", linewidth=0.5)

    for i, (val, label) in enumerate(zip(top_types.values, top_types.index)):
        ax.text(val + 5, i, f"{int(val):,}", va="center",
                fontsize=8.5, color="#ccccff")

    ax.set_title("Most Common Incident Types (FEMA)",
                 fontsize=15, fontweight="bold", pad=14)
    ax.set_xlabel("Number of Declarations", fontsize=11)
    ax.set_ylabel("Incident Type", fontsize=11)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.grid(axis="x")
    plt.tight_layout()
    plt.savefig("plot_incident_types.png", dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print("  Saved: plot_incident_types.png")

print("\n✓  All steps complete.")