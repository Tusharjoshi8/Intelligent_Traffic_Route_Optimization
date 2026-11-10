"""Evaluate traffic prediction models."""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from xgboost import XGBRegressor


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "traffic"
    / "processed"
    / "model_data"
)

MODEL_DIR = PROJECT_ROOT / "machine_learning" / "models"

TEST_PATH = DATA_DIR / "test_data.csv"
MODEL_PATH = MODEL_DIR / "traffic_model.json"
METADATA_PATH = MODEL_DIR / "model_metadata.json"


def evaluate_traffic_model():

    print("\nXGBOOST MODEL EVALUATION")
    print("====================================")

    # Step 1: Load testing dataset
    test_data = pd.read_csv(TEST_PATH)

    X_test = test_data.drop(columns=["traffic_volume"])
    y_test = test_data["traffic_volume"]

    # Step 2: Load trained model
    model = XGBRegressor()
    model.load_model(str(MODEL_PATH))

    # Step 3: Generate predictions
    predictions = model.predict(X_test)

    # Step 4: Calculate evaluation metrics
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    # Step 5: Display results
    print("\nEVALUATION RESULTS")
    print("------------------------------------")

    print("Mean Absolute Error (MAE):", round(mae, 2))
    print("Root Mean Squared Error (RMSE):", round(rmse, 2))
    print("R2 Score:", round(r2, 4))
    print("R2 Percentage:", round(r2 * 100, 2), "%")

    # Step 6: Show actual vs predicted values
    results = pd.DataFrame({
        "Actual Traffic Volume": y_test.values,
        "Predicted Traffic Volume": predictions.round(2)
    })

    print("\nACTUAL VS PREDICTED TRAFFIC")
    print("------------------------------------")
    print(results.head(10).to_string(index=False))

    # Step 7: Save evaluation results
    evaluation = {
        "MAE": round(float(mae), 2),
        "RMSE": round(float(rmse), 2),
        "R2_Score": round(float(r2), 4),
        "R2_Percentage": round(float(r2 * 100), 2)
    }

    with open(MODEL_DIR / "evaluation_results.json", "w") as file:
        json.dump(evaluation, file, indent=4)

    print("\nEvaluation results saved successfully!")


if __name__ == "__main__":
    evaluate_traffic_model()