from pathlib import Path
from mldesigner import command_component, Input, Output

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

import mlflow
from mlflow.tracking import MlflowClient

@command_component(
    name="training",
    display_name="Training",
    description="Trains the model",
    environment=dict(
        conda_file=Path(__file__).parent / "conda.yaml",
        image="mcr.microsoft.com/azureml/curated/sklearn-1.1:17",
    ),
)

def training_component(
    X_train_input: Input(type='uri_file'),
    y_train_input: Input(type='uri_file'),
    model_output: Output(type='uri_file'),
    run_id_output: Output(type='uri_file'),
):

    # Set names
    model_name = "MC-ResponsePredictor"

    # Read data from input paths
    X_train = pd.read_csv(X_train_input, sep=';')
    y_train = pd.read_csv(y_train_input, sep=';')

    # Initialize model
    rf_model = RandomForestClassifier(n_estimators = 100, random_state=42)

    # Start MLflow run
    with mlflow.start_run():

        # Fit model
        rf_model.fit(X_train, y_train)

        # Log model params
        mlflow.log_params(rf_model.get_params())

        # Save model to the output path
        joblib.dump(rf_model, model_output)

        # Log the model
        mlflow.sklearn.log_model(rf_model, "model")

        # Register the model
        mlflow.register_model(
            "runs:/{}/model".format(mlflow.active_run().info.run_id), model_name)
        
        # Save mlflow run id
        run_id_df = pd.DataFrame({"run_id":[mlflow.active_run().info.run_id]})
        run_id_df.to_csv(run_id_output, index=False)

    
