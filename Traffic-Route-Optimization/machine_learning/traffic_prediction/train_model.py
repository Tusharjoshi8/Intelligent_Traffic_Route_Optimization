"""Train traffic prediction models."""
import pandas as pd
import json
import time
from pathlib import Path

from xgboost_model import create_xgboost_model


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "traffic"
    / "processed"
    / "model_data"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "machine_learning"
    / "models"
)

TRAIN_PATH = DATA_DIR / "train_data.csv"
TEST_PATH = DATA_DIR / "test_data.csv"

MODEL_PATH = MODEL_DIR / "traffic_model.json"
METADATA_PATH = MODEL_DIR / "model_metadata.json"


def train_traffic_model():

    print("\nXGBOOST MODEL TRAINING")
    print("====================================")

    # Step 1: Load training and testing datasets
    train_data = pd.read_csv(TRAIN_PATH)
    test_data = pd.read_csv(TEST_PATH)

    print("Training dataset shape:", train_data.shape)
    print("Testing dataset shape:", test_data.shape)

    # Step 2: Separate input features and target
    target_column = "traffic_volume"

    X_train = train_data.drop(columns=[target_column])
    y_train = train_data[target_column]

    X_test = test_data.drop(columns=[target_column])
    y_test = test_data[target_column]

    # Step 3: Create XGBoost model
    model = create_xgboost_model()

    print("\nTraining started...")

    # Step 4: Train the model
    start_time = time.time()

    model.fit(X_train, y_train)

    training_time = time.time() - start_time

    print("\nMODEL TRAINING COMPLETED!")
    print("====================================")

    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))
    print("Number of features:", X_train.shape[1])
    print("Training time:", round(training_time, 2), "seconds")

    # Step 5: Create model directory
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Step 6: Save trained model
    model.save_model(str(MODEL_PATH))

    # Step 7: Save model metadata
    metadata = {
        "model_name": "XGBoost Regressor",
        "target": target_column,
        "features": X_train.columns.tolist(),
        "training_records": len(X_train),
        "testing_records": len(X_test),
        "number_of_features": X_train.shape[1],
        "training_time_seconds": round(training_time, 2)
    }

    with open(METADATA_PATH, "w") as file:
        json.dump(metadata, file, indent=4)

    print("\nMODEL SAVED SUCCESSFULLY!")
    print("Model path:", MODEL_PATH)
    print("Metadata path:", METADATA_PATH)


if __name__ == "__main__":
    train_traffic_model()