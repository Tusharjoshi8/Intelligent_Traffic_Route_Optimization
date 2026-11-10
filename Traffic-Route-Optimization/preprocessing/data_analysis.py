import pandas as pd
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "traffic"
    / "raw"
    / "Metro_Interstate_Traffic_Volume.csv"
)


def analyze_traffic_data():

    print("TRAFFIC DATA ANALYSIS")
    print("==============================")

    # Load dataset
    data = pd.read_csv(DATA_PATH)

    # 1. Dataset information
    print("\n1. DATASET INFORMATION")
    print("------------------------------")
    print("Shape:", data.shape)
    print("\nData Types:")
    print(data.dtypes)

    # 2. Missing values
    print("\n2. MISSING VALUES")
    print("------------------------------")
    print(data.isnull().sum())

    # 3. Duplicate records
    print("\n3. DUPLICATE RECORDS")
    print("------------------------------")
    print(data.duplicated().sum())

    # 4. Traffic volume statistics
    print("\n4. TRAFFIC VOLUME STATISTICS")
    print("------------------------------")
    print(data["traffic_volume"].describe())

    # 5. Date range
    print("\n5. DATE RANGE")
    print("------------------------------")

    data["date_time"] = pd.to_datetime(
        data["date_time"],
        errors="coerce"
    )

    print("Start:", data["date_time"].min())
    print("End:", data["date_time"].max())

    # 6. Weather conditions
    print("\n6. WEATHER CONDITIONS")
    print("------------------------------")
    print(data["weather_main"].value_counts())

    # 7. Traffic volume by hour
    print("\n7. TRAFFIC VOLUME BY HOUR")
    print("------------------------------")

    data["hour"] = data["date_time"].dt.hour

    print(
        data.groupby("hour")["traffic_volume"].mean().round(2)
    )

    print("\nANALYSIS COMPLETED!")


if __name__ == "__main__":
    analyze_traffic_data()