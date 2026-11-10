import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from xgboost import XGBRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "machine_learning"
    / "models"
    / "traffic_model.json"
)


def analyze_feature_importance():

    print("\nXGBOOST FEATURE IMPORTANCE")
    print("====================================")

    # Load trained model
    model = XGBRegressor()
    model.load_model(str(MODEL_PATH))

    # Get feature importance
    importance = model.feature_importances_

    feature_names = model.get_booster().feature_names

    # Create DataFrame
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    })

    # Sort from highest to lowest
    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    print("\nTOP 10 IMPORTANT FEATURES")
    print("------------------------------------")

    print(
        importance_df.head(10).to_string(index=False)
    )

    # Plot feature importance
    plt.figure(figsize=(10, 6))

    top_features = importance_df.head(10).sort_values(
        by="Importance"
    )

    plt.barh(
        top_features["Feature"],
        top_features["Importance"]
    )

    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    plt.title("Top 10 XGBoost Feature Importances")

    plt.tight_layout()

    # Save chart
    output_path = (
        PROJECT_ROOT
        / "machine_learning"
        / "models"
        / "feature_importance.png"
    )

    plt.savefig(output_path, dpi=300)

    print("\nFeature importance chart saved at:")
    print(output_path)

    plt.show()


if __name__ == "__main__":
    analyze_feature_importance()