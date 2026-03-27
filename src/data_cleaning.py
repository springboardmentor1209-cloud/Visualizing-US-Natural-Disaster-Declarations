import pandas as pd


def clean_dataset(input_file, output_file):

    # Load dataset
    df = pd.read_csv(input_file)

    print("Original Shape:", df.shape)

    # -----------------------------
    # 1. Standardize column names
    # -----------------------------
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("&", "and")
    )

    # -----------------------------
    # 2. Remove duplicates
    # -----------------------------
    df = df.drop_duplicates()

    # -----------------------------
    # 3. Convert date columns
    # -----------------------------
    date_columns = [
        "declaration_date",
        "start_date",
        "end_date",
        "close_date"
    ]

    for col in date_columns:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .replace({"": None})
            )

            df[col] = pd.to_datetime(
                df[col],
                errors="coerce",
                format="mixed"
            )

    # -----------------------------
    # 4. Handle missing values
    # -----------------------------
    if "county" in df.columns:
        df["county"] = df["county"].fillna("Unknown")

    df = df.dropna(subset=["state", "disaster_type", "declaration_date"])

    # -----------------------------
    # 5. Standardize categorical fields
    # -----------------------------
    df["state"] = df["state"].str.strip().str.upper()

    df["disaster_type"] = df["disaster_type"].str.strip().str.title()

    df["declaration_type"] = df["declaration_type"].str.strip().str.title()

    # -----------------------------
    # 6. Convert program columns
    # -----------------------------
    program_columns = [
        "individual_assistance_program",
        "individuals_and_households_program",
        "public_assistance_program",
        "hazard_mitigation_program"
    ]

    for col in program_columns:
        if col in df.columns:
            df[col] = df[col].map({
                "Yes": 1,
                "No": 0
            })

    # -----------------------------
    # 7. Feature Engineering
    # -----------------------------
    df["year"] = df["declaration_date"].dt.year
    df["month"] = df["declaration_date"].dt.month
    df["quarter"] = df["declaration_date"].dt.quarter

    # Incident duration
    if "start_date" in df.columns and "end_date" in df.columns:
        df["disaster_duration_days"] = (
            df["end_date"] - df["start_date"]
        ).dt.days

        df = df[df["disaster_duration_days"] >= 0]

    # -----------------------------
    # 8. Reset index
    # -----------------------------
    df = df.reset_index(drop=True)

    print("Cleaned Shape:", df.shape)

    # -----------------------------
    # 9. Save cleaned dataset
    # -----------------------------
    df.to_csv(output_file, index=False)

    print("Cleaned dataset saved to:", output_file)

    return df