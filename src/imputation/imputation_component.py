from pathlib import Path
from mldesigner import command_component, Input, Output

import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import os
import joblib

import mlflow
from mlflow.tracking import MlflowClient

@command_component(
    name="imputation",
    display_name="Imputation",
    description="Imputes the missing values",
    environment=dict(
        conda_file=Path(__file__).parent / "conda.yaml",
        image="mcr.microsoft.com/azureml/curated/sklearn-1.1:17",
    ),
)

def imputation_component(
    X_train_input: Input(type='uri_file'),
    X_test_input: Input(type='uri_file'),
    X_train_output: Output(type='uri_file'),
    X_test_output: Output(type='uri_file'),
    run_id_output: Output(type='uri_file'),
):
    # Read data from the input paths
    X_train = pd.read_csv(X_train_input, sep=';')
    X_test = pd.read_csv(X_test_input, sep=';')

    # Select the numeric columns
    numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()

    # Initialize the imputer
    iter_imputer = IterativeImputer(random_state=42)

    # Fit the imputer with the training data
    iter_imputer.fit(X_train[numeric_cols])

    # Save the fitted imputer as an artifact
    with mlflow.start_run():
        # Define the path to save the fitted imputer
        directory = './artifacts-' + str(mlflow.active_run().info.run_id) + '/'
        filename = 'imputer.pkl'
        imputer_path = os.path.join(directory, filename)

        # Check if the directory exists, if not create it
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Save the fitted imputer
        joblib.dump(iter_imputer, imputer_path)

        # Log the fitted imputer as an artifact
        mlflow.log_artifact(imputer_path)

        # Save mlflow run id
        run_id_df = pd.DataFrame({"run_id":[mlflow.active_run().info.run_id]})
        run_id_df.to_csv(run_id_output, index=False)
    
    # Impute the data
    X_train[numeric_cols] = iter_imputer.transform(X_train[numeric_cols])
    X_test[numeric_cols] = iter_imputer.transform(X_test[numeric_cols])

    # Saving to output paths
    X_train.to_csv(X_train_output, index=False, sep=';')
    X_test.to_csv(X_test_output, index=False, sep=';')

