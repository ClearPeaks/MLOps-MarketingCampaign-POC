from pathlib import Path
from mldesigner import command_component, Input, Output

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

import mlflow
from mlflow.tracking import MlflowClient

@command_component(
    name="normalization",
    display_name="Normalization",
    description="Normalizes the data.",
    environment=dict(
        conda_file=Path(__file__).parent / "conda.yaml",
        image="mcr.microsoft.com/azureml/curated/sklearn-1.1:17",
    ),
)

def normalization_component(
    X_train_input: Input(type='uri_file'),
    X_test_input: Input(type='uri_file'),
    X_train_output: Output(type='uri_file'),
    X_test_output: Output(type='uri_file'),
    run_id_input: Input(type='uri_file'),
    run_id_output: Output(type='uri_file'),
):
    # Read data from input paths
    X_train = pd.read_csv(X_train_input, sep=';')
    X_test = pd.read_csv(X_test_input, sep=';')
    run_id = pd.read_csv(run_id_input)
    run_id = run_id['run_id'][0]

    # Select numeric columns
    numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()
    
    # Create the scaler
    scaler = StandardScaler()

    # Fit the scaler with the training data
    scaler.fit(X_train[numeric_cols])

    # Save the scaler as an artifact
    with mlflow.start_run():
        # Define the path to save the fitted scaler
        directory = './artifacts-' + str(mlflow.active_run().info.run_id) + '/'
        filename = 'scaler.pkl'
        scaler_path = os.path.join(directory, filename)

        # Check if the directory exists, if not create it
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Save the fitted scaler
        joblib.dump(imp, scaler_path)

        # Log the fitted scaler as an artifact
        mlflow.log_artifact(scaler_path)

        # Save mlflow run id
        run_id_df = pd.DataFrame({"run_id":[mlflow.active_run().info.run_id]})
        run_id_df.to_csv(run_id_output, index=False)
    
    # Normalize the data
    X_train[numeric_cols] = scaler.transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    # Saving to output paths
    X_train.to_csv(X_train_output, index=False, sep=';')
    X_test.to_csv(X_test_output, index=False, sep=';')
    