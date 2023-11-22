from pathlib import Path
from mldesigner import command_component, Input, Output

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

@command_component(
    name="normalization",
    version="1",
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
):
    # Read data from input paths
    X_train = pd.read_csv(X_train_input, sep=';')
    X_test = pd.read_csv(X_test_input, sep=';')

    # Select numeric columns
    numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()
    
    # Create the scaler
    scaler = StandardScaler()

    # Fit the scaler with the training data
    scaler.fit(X_train[numeric_cols])
    
    # Normalize the data
    X_train[numeric_cols] = scaler.transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    # Saving to output paths
    X_train.to_csv(X_train_output, index=False, sep=';')
    X_test.to_csv(X_test_output, index=False, sep=';')
    