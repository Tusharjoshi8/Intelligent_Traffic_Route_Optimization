
import pandas as pd
from pathlib import Path


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "traffic"
    / "processed"
    / "engineered_traffic_data.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "traffic"
    / "processed"
    / "model_data"
)


def prepare_traffic_data():

    print("\nTRAFFIC MODEL DATA PREPARATION")
    print("====================================")

    # Step 1: Load engineered dataset
    data = pd.read_csv(INPUT_PATH)

    print("Original dataset shape:", data.shape)

    # Step 2: Convert date_time into datetime format
    data["date_time"] = pd.to_datetime(data["date_time"])

    # Step 3: Sort records chronologically
    data = data.sort_values("date_time").reset_index(drop=True)

    # Step 4: Select input features
    selected_features = [
        "temp",
        "rain_1h",
        "snow_1h",
        "clouds_all",
        "weather_main",
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
    ]

    X = data[selected_features]

    # Step 5: Select prediction target
    y = data["traffic_volume"]

    print("\nInput features:", X.shape)
    print("Prediction target:", y.shape)

    # Step 6: Split chronologically (80% training, 20% testing)
    split_index = int(len(data) * 0.80)

    X_train = X.iloc[:split_index].copy()
    X_test = X.iloc[split_index:].copy()

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    # Step 7: Encode categorical weather column
    X_train = pd.get_dummies(
        X_train,
        columns=["weather_main"],
        dtype=int
    )

    X_test = pd.get_dummies(
        X_test,
        columns=["weather_main"],
        dtype=int
    )

    # Ensure training and testing have identical columns
    X_test = X_test.reindex(
        columns=X_train.columns,
        fill_value=0
    )

    # Step 8: Combine features and target
    train_data = X_train.copy()
    train_data["traffic_volume"] = y_train.values

    test_data = X_test.copy()
    test_data["traffic_volume"] = y_test.values

    # Step 9: Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 10: Save training dataset
    train_data.to_csv(
        OUTPUT_DIR / "train_data.csv",
        index=False
    )

    # Step 11: Save testing dataset
    test_data.to_csv(
        OUTPUT_DIR / "test_data.csv",
        index=False
    )

    print("\nDATA PREPARATION COMPLETED!")
    print("====================================")

    print("Training records:", len(train_data))
    print("Testing records:", len(test_data))

    print("\nTraining dataset shape:", train_data.shape)
    print("Testing dataset shape:", test_data.shape)

    print("\nTraining period:")
    print(data["date_time"].iloc[0], "to",
          data["date_time"].iloc[split_index - 1])

    print("\nTesting period:")
    print(data["date_time"].iloc[split_index], "to",
          data["date_time"].iloc[-1])

    print("\nFiles saved in:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    prepare_traffic_data()
