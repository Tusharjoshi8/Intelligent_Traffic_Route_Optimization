"""Prepare traffic data for analysis and modeling."""
from pathlib import Path

from dataset_loader import load_dataset
from data_cleaning import clean_dataset
from feature_engineering import engineer_features



def preprocess_traffic_dataset(file_name, timestamp_column):
    """
    Complete traffic dataset preprocessing pipeline.

    Steps:
    1. Load dataset
    2. Clean dataset
    3. Perform feature engineering
    4. Save processed dataset
    """

    print("\nTRAFFIC DATA PREPROCESSING PIPELINE")
    print("====================================")

    # Step 1: Load dataset
    print("\nSTEP 1: Loading Dataset")

    data = load_dataset(file_name)

    # Step 2: Clean dataset
    print("\nSTEP 2: Cleaning Dataset")

    data = clean_dataset(data)

    # Step 3: Feature engineering
    print("\nSTEP 3: Feature Engineering")

    data = engineer_features(data, timestamp_column)

    # Step 4: Save processed dataset
    project_root = Path(__file__).resolve().parent.parent

    processed_dir = project_root / "datasets" / "traffic" / "processed"

    processed_dir.mkdir(parents=True, exist_ok=True)

    output_path = processed_dir / f"processed_{file_name}"

    data.to_csv(output_path, index=False)

    print("\nPREPROCESSING COMPLETED!")
    print("====================================")
    print("Final dataset shape:", data.shape)
    print("Processed dataset saved at:")
    print(output_path)

    return data


if __name__ == "__main__":

    print("TRAFFIC PREPROCESSING SYSTEM")

    file_name = input(
        "\nEnter traffic CSV filename: "
    )

    timestamp_column = input(
        "Enter timestamp column name: "
    )

    try:

        processed_data = preprocess_traffic_dataset(
            file_name,
            timestamp_column
        )

    except (FileNotFoundError, ValueError, KeyError) as error:

        print("\nPreprocessing Error:")
        print(error)