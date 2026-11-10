"""Predict traffic volume using a trained model."""
import pandas as pd
from pathlib import Path
from xgboost import XGBRegressor


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "traffic"
    / "processed"
    / "model_data"
    / "test_data.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "machine_learning"
    / "models"
    / "traffic_model.json"
)


def predict_traffic_volume():

    print("\nTRAFFIC VOLUME PREDICTION")
    print("====================================")

    # Step 1: Load testing data
    data = pd.read_csv(DATA_PATH)

    # Step 2: Separate features and actual traffic volume
    X = data.drop(columns=["traffic_volume"])
    actual_values = data["traffic_volume"]

    # Step 3: Load trained model
    model = XGBRegressor()
    model.load_model(str(MODEL_PATH))

    # Step 4: Select one test record
    sample = X.iloc[[0]]

    # Step 5: Predict traffic volume
    prediction = model.predict(sample)[0]

    actual = actual_values.iloc[0]

    # Step 6: Display prediction
    print("\nPREDICTION RESULT")
    print("------------------------------------")

    print("Actual Traffic Volume:", actual)
    print("Predicted Traffic Volume:", round(float(prediction), 2))

    print(
        "Absolute Prediction Error:",
        round(abs(actual - prediction), 2)
    )

    print("\nPREDICTION COMPLETED!")


if __name__ == "__main__":
    predict_traffic_volume()