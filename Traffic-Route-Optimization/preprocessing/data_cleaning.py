import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "traffic"
    / "raw"
    / "Metro_Interstate_Traffic_Volume.csv"
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "traffic"
    / "processed"
)


def clean_traffic_dataset():

    print("\nTRAFFIC DATA CLEANING")
    print("==============================")

    # Step 1: Load original dataset
    data = pd.read_csv(RAW_DATA_PATH)

    print("Original dataset shape:", data.shape)

    # Step 2: Convert date_time into datetime format
    data["date_time"] = pd.to_datetime(
        data["date_time"],
        errors="coerce"
    )

    # Step 3: Remove records with invalid timestamps
    data = data.dropna(subset=["date_time"])

    # Step 4: Remove duplicate records
    duplicate_count = data.duplicated().sum()

    data = data.drop_duplicates()

    print("Duplicates removed:", duplicate_count)

    # Step 5: Handle holiday information
    # Missing holiday values mean no recorded holiday.
    data["holiday"] = data["holiday"].fillna("None")

    # Step 6: Handle missing numeric values
    numeric_columns = [
        "temp",
        "rain_1h",
        "snow_1h",
        "clouds_all"
    ]

    for column in numeric_columns:

        median_value = data[column].median()

        data[column] = data[column].fillna(median_value)

    # Step 7: Handle missing weather categories
    categorical_columns = [
        "weather_main",
        "weather_description"
    ]

    for column in categorical_columns:

        data[column] = data[column].fillna("Unknown")

    # Step 8: Remove records with missing prediction target
    data = data.dropna(subset=["traffic_volume"])

    # Step 9: Sort records chronologically
    data = data.sort_values("date_time")

    # Step 10: Reset row indexes
    data = data.reset_index(drop=True)

    # Step 11: Save cleaned dataset
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    output_path = (
        PROCESSED_DIR / "cleaned_traffic_data.csv"
    )

    data.to_csv(output_path, index=False)

    print("\nCLEANING COMPLETED!")
    print("==============================")

    print("Final dataset shape:", data.shape)

    print("\nRemaining missing values:")
    print(data.isnull().sum())

    print("\nCleaned dataset saved at:")
    print(output_path)

    return data


if __name__ == "__main__":

    cleaned_data = clean_traffic_dataset()