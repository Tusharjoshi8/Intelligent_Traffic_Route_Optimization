"""XGBoost model definitions for traffic prediction."""
from xgboost import XGBRegressor


def create_xgboost_model():
    """
    Create and configure the XGBoost regression model
    for traffic volume prediction.
    """

    model = XGBRegressor(

        # Number of decision trees
        n_estimators=400,

        # Learning speed
        learning_rate=0.05,

        # Maximum depth of each tree
        max_depth=8,

        # Percentage of training samples used per tree
        subsample=0.8,

        # Percentage of features used per tree
        colsample_bytree=0.8,

        # Regression objective
        objective="reg:squarederror",

        # Evaluation metric
        eval_metric="rmse",

        # Reproducible results
        random_state=42,

        # Use available CPU cores
        n_jobs=-1
    )

    return model


if __name__ == "__main__":

    model = create_xgboost_model()

    print("XGBOOST MODEL CREATED SUCCESSFULLY!")
    print("------------------------------------")
    print(model)