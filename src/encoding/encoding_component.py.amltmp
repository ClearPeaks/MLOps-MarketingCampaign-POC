from pathlib import Path
from mldesigner import command_component, Input, Output

import pandas as pd

@command_component(
    name="encoding",
    display_name="Encoding",
    description="Performs the encoding of the categorical variables in the dataset.",
    environment=dict(
        conda_file=Path(__file__).parent / "conda.yaml",
        image="mcr.microsoft.com/azureml/curated/sklearn-1.1:17",
    ),
    is_deterministic = False
)

def encoding_component(
    X_train_input: Input(type='uri_file'),
    X_test_input: Input(type='uri_file'),
    X_train_output: Output(type='uri_file'),
    X_test_output: Output(type='uri_file'),
):
    # Read data from input paths
    X_train = pd.read_csv(X_train_input, sep=';')
    X_test = pd.read_csv(X_test_input, sep=';')

    # One-hot encoding for categorical variables
    X_train = pd.get_dummies(X_train, columns=['Education', 'Marital_Status'], prefix=['Edu', 'MS'])
    X_test = pd.get_dummies(X_test, columns=['Education', 'Marital_Status'], prefix=['Edu', 'MS'])

    # Saving to output paths
    X_train.to_csv(X_train_output, index=False, sep=';')
    X_test.to_csv(X_test_output, index=False, sep=';')