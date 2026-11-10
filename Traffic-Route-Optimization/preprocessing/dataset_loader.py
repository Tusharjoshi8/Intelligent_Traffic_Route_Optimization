# Import required libraries

from pathlib import Path
import pandas as pd


# Find the main project directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# Define the raw traffic dataset directory
RAW_DATA_DIR = PROJECT_ROOT / "datasets" / "traffic" / "raw"


def load_dataset(file_name):
    """
    Load a traffic CSV dataset using Pandas.

    Args:
        file_name: Name of the CSV dataset.

    Returns:
        pandas DataFrame containing traffic data.
    """

    # Create the complete dataset path
    file_path = RAW_DATA_DIR / file_name

    # Check whether the dataset exists
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    # Check the file format
    if file_path.suffix.lower() != ".csv":
        raise ValueError("Only CSV files are supported.")

    # Read CSV file using Pandas
    data = pd.read_csv(file_path)

    # Display dataset information
    print("\nDATASET LOADED SUCCESSFULLY!")
    print("--------------------------------")

    print("\nDataset Shape:")
    print(data.shape)

    print("\nColumn Names:")
    print(data.columns.tolist())

    print("\nFirst Five Records:")
    print(data.head())

    return data


# Execute the file directly
if __name__ == "__main__":

    print("TRAFFIC DATASET LOADER")
    print("--------------------------------")

    dataset_name = input(
        "Enter CSV dataset filename: "
    )

    try:
        traffic_data = load_dataset(dataset_name)

    except (FileNotFoundError, ValueError) as error:
        print(f"\nError: {error}")