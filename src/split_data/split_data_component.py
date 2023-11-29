from pathlib import Path
from mldesigner import command_component, Input, Output

import pandas as pd
from sklearn.model_selection import train_test_split

@command_component(
    name="split_data",
    display_name="Split Data",
    description="Performs the train-test split of the data.",
    environment=dict(
        conda_file=Path(__file__).parent / "conda.yaml",
        image="mcr.microsoft.com/azureml/curated/sklearn-1.1:17",
    ),
)

def split_data_component(
    input_data: Input(type='uri_file'),
    X_train_data: Output(type='uri_file'),
    X_test_data: Output(type='uri_file'),
    y_train_data: Output(type='uri_file'),
    y_test_data: Output(type='uri_file'),
):
    # Read data from input path
    data = pd.read_csv(input_data, sep=';')

    # Separating the data from the response
    X = data.drop(['Response'], axis=1)
    y = data['Response']
    
    # Splitting the data with a fixed seed for reproducibility
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=data['Response'])

    # Saving to output paths
    X_train.to_csv(X_train_data, index=False, sep=';')
    X_test.to_csv(X_test_data, index=False, sep=';')
    y_train.to_csv(y_train_data, index=False, sep=';')
    y_test.to_csv(y_test_data, index=False, sep=';')

    