# Import required libraries

from pathlib import Path
import shutil


# Find the main project directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# Define the raw traffic dataset directory
RAW_DATA_DIR = PROJECT_ROOT / "datasets" / "traffic" / "raw"


def collect_dataset(source_path):
    """
    Copy a traffic dataset into the project's raw dataset folder.
    """

    # Convert the given path into a Path object
    source_path = Path(source_path).expanduser()

    # Check whether the source file exists
    if not source_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {source_path}"
        )

    # Check whether the file is CSV
    if source_path.suffix.lower() != ".csv":
        raise ValueError("Only CSV datasets are supported.")

    # Create the destination folder if it doesn't exist
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Define destination path
    destination = RAW_DATA_DIR / source_path.name

    # Copy the dataset
    shutil.copy2(source_path, destination)

    print("Dataset collected successfully!")
    print(f"Saved at: {destination}")

    return destination


# Run this file directly
if __name__ == "__main__":

    print("TRAFFIC DATA COLLECTION MODULE")
    print("--------------------------------")

    print("Raw dataset folder:")
    print(RAW_DATA_DIR)

    print("\nReady to collect traffic datasets.")