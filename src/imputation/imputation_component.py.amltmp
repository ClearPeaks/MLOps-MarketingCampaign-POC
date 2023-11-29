from pathlib import Path
from mldesigner import command_component, Input, Output

import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

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
):
    # Read data from input paths
    X_train = pd.read_csv(X_train_input, sep=';')
    X_test = pd.read_csv(X_test_input, sep=';')

    # Select numeric columns
    numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()

    # Initialize the imputer
    iter_imputer = IterativeImputer(random_state=42)

    # Fit the imputer with the training data
    iter_imputer.fit(X_train[numeric_cols])
    
    # Impute the data
    X_train[numeric_cols] = iter_imputer.transform(X_train[numeric_cols])
    X_test[numeric_cols] = iter_imputer.transform(X_test[numeric_cols])

    # Saving to output paths
    X_train.to_csv(X_train_output, index=False, sep=';')
    X_test.to_csv(X_test_output, index=False, sep=';')

