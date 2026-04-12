import pandas as pd
import numpy as np
import json
import os
import sys
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

NAME = "Kaustubh Bhalerao"
ROLL_NO = "2022bcs0172"

# Get run config from environment variables
RUN_NUMBER     = int(os.environ.get("RUN_NUMBER", 1))
MODEL_TYPE     = os.environ.get("MODEL_TYPE", "RandomForest")
N_ESTIMATORS   = int(os.environ.get("N_ESTIMATORS", 100))
MAX_DEPTH      = os.environ.get("MAX_DEPTH", "None")
MAX_DEPTH      = None if MAX_DEPTH == "None" else int(MAX_DEPTH)
USE_ALL_FEATURES = os.environ.get("USE_ALL_FEATURES", "true").lower() == "true"
DATASET_VERSION = os.environ.get("DATASET_VERSION", "v1")

# Load data
df = pd.read_csv("data/housing.csv")
print(f"Dataset size: {len(df)} rows")

# Features and target
ALL_FEATURES = [
    "longitude", "latitude", "housing_median_age",
    "total_rooms", "total_bedrooms", "population",
    "households", "median_income"
]

REDUCED_FEATURES = [
    "median_income", "housing_median_age",
    "total_rooms", "households"
]

if USE_ALL_FEATURES:
    features = ALL_FEATURES
    feature_set = "all_features"
else:
    features = REDUCED_FEATURES
    feature_set = "reduced_features"

X = df[features]
y = df["median_house_value"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training samples: {len(X_train)}")
print(f"Features used: {features}")

# MLflow setup
mlflow.set_experiment(f"{ROLL_NO}_experiment")

with mlflow.start_run(run_name=f"Run_{RUN_NUMBER}_{MODEL_TYPE}_{DATASET_VERSION}"):

    # Log parameters
    mlflow.log_param("run_number", RUN_NUMBER)
    mlflow.log_param("model_type", MODEL_TYPE)
    mlflow.log_param("n_estimators", N_ESTIMATORS)
    mlflow.log_param("max_depth", MAX_DEPTH)
    mlflow.log_param("feature_set", feature_set)
    mlflow.log_param("features_used", str(features))
    mlflow.log_param("dataset_version", DATASET_VERSION)
    mlflow.log_param("dataset_size", len(df))
    mlflow.log_param("training_samples", len(X_train))
    mlflow.log_param("name", NAME)
    mlflow.log_param("roll_no", ROLL_NO)

    # Train model
    if MODEL_TYPE == "RandomForest":
        model = RandomForestRegressor(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            random_state=42
        )
    elif MODEL_TYPE == "GradientBoosting":
        model = GradientBoostingRegressor(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH if MAX_DEPTH else 3,
            random_state=42
        )
    else:
        model = LinearRegression()

    model.fit(X_train, y_train)

    # Evaluate
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print(f"RMSE: {rmse:.4f}")
    print(f"R2: {r2:.4f}")

    # Log metrics
    mlflow.log_metric("rmse", round(rmse, 4))
    mlflow.log_metric("r2", round(r2, 4))

    # Log model
    mlflow.sklearn.log_model(model, "model")

    # Save metrics JSON
    os.makedirs("app/artifacts", exist_ok=True)
    metrics = {
        "name": NAME,
        "roll_no": ROLL_NO,
        "run_number": RUN_NUMBER,
        "model_type": MODEL_TYPE,
        "dataset_version": DATASET_VERSION,
        "dataset_size": len(df),
        "training_samples": len(X_train),
        "feature_set": feature_set,
        "features_used": features,
        "rmse": round(rmse, 4),
        "r2": round(r2, 4)
    }

    with open("app/artifacts/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    mlflow.log_artifact("app/artifacts/metrics.json")

    print(f"Run {RUN_NUMBER} complete - RMSE: {rmse:.4f}, R2: {r2:.4f}")
