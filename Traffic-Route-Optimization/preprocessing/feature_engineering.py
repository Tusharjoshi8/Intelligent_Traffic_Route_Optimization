import pandas as pd
import numpy as np
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "traffic"
    / "processed"
    / "cleaned_traffic_data.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "traffic"
    / "processed"
)


def extract_time_features(data):

    data = data.copy()

    # Convert date_time into datetime format
    data["date_time"] = pd.to_datetime(data["date_time"])

    # Extract time features
    data["hour"] = data["date_time"].dt.hour

    data["day_of_week"] = data["date_time"].dt.dayofweek

    data["day"] = data["date_time"].dt.day

    data["month"] = data["date_time"].dt.month

    # Weekend identification
    data["is_weekend"] = (
        data["day_of_week"] >= 5
    ).astype(int)

    # Morning and evening peak hours
    data["is_peak_hour"] = (
        ((data["hour"] >= 7) & (data["hour"] <= 10)) |
        ((data["hour"] >= 17) & (data["hour"] <= 20))
    ).astype(int)

    print("Time features created successfully!")

    return data


def create_cyclical_features(data):

    data = data.copy()

    # Hour repeats every 24 hours
    data["hour_sin"] = np.sin(
        2 * np.pi * data["hour"] / 24
    )

    data["hour_cos"] = np.cos(
        2 * np.pi * data["hour"] / 24
    )

    # Day of week repeats every 7 days
    data["day_sin"] = np.sin(
        2 * np.pi * data["day_of_week"] / 7
    )

    data["day_cos"] = np.cos(
        2 * np.pi * data["day_of_week"] / 7
    )

    print("Cyclical features created successfully!")

    return data


def create_holiday_feature(data):

    data = data.copy()

    data["is_holiday"] = (
        data["holiday"].notna() &
        (data["holiday"] != "None")
    ).astype(int)

    print("Holiday feature created successfully!")

    return data


def engineer_traffic_features():

    print("\nTRAFFIC FEATURE ENGINEERING")
    print("==============================")

    # Step 1: Load cleaned dataset
    data = pd.read_csv(INPUT_PATH)

    print("Original dataset shape:", data.shape)

    # Step 2: Extract time features
    data = extract_time_features(data)

    # Step 3: Create cyclical features
    data = create_cyclical_features(data)

    # Step 4: Create holiday feature
    data = create_holiday_feature(data)

    # Step 5: Sort records chronologically
    data = data.sort_values("date_time")

    data = data.reset_index(drop=True)

    # Step 6: Save engineered dataset
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = (
        OUTPUT_DIR / "engineered_traffic_data.csv"
    )

    data.to_csv(output_path, index=False)

    print("\nFEATURE ENGINEERING COMPLETED!")
    print("==============================")

    print("Final dataset shape:", data.shape)

    print("\nNew Features:")
    print([
        "hour",
        "day_of_week",
        "day",
        "month",
        "is_weekend",
        "is_peak_hour",
        "hour_sin",
        "hour_cos",
        "day_sin",
        "day_cos",
        "is_holiday"
    ])

    print("\nEngineered dataset saved at:")
    print(output_path)

    return data


if __name__ == "__main__":

    engineered_data = engineer_traffic_features()