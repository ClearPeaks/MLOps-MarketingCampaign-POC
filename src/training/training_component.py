from pathlib import Path
from mldesigner import command_component, Input, Output

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

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
):
    # Read data from input paths
    X_train = pd.read_csv(X_train_input, sep=';')
    y_train = pd.read_csv(y_train_input, sep=';')

    # Initialize model
    rf_model = RandomForestClassifier(n_estimators = 100, random_state=42)

    # Fit model
    rf_model.fit(X_train, y_train)

    # Save model to the output path
    joblib.dump(rf_model, model_output)

    
