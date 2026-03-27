from src.data_cleaning import clean_dataset


def main():

    input_file = "data/database.csv"
    output_file = "data/cleaned_disaster_data.csv"

    clean_df = clean_dataset(input_file, output_file)

    print("\nCleaned Dataset Info:\n")
    print(clean_df.info())


if __name__ == "__main__":
    main()