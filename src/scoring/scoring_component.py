from pathlib import Path
from mldesigner import command_component, Input, Output

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

import mlflow
from mlflow.tracking import MlflowClient

@command_component(
    name="scoring",
    display_name="Scoring",
    description="Tests the performance of the model.",
    environment=dict(
        conda_file=Path(__file__).parent / "conda.yaml",
        image="mcr.microsoft.com/azureml/curated/sklearn-1.1:17",
    ),
    is_deterministic = False
)

def scoring_component(
    X_test_input: Input(type='uri_file'),
    y_test_input: Input(type='uri_file'),
    model_input: Input(type='uri_file'),
    run_id_input: Input(type='uri_file'),
    predictions_output: Output(type='uri_file'),
):
    # Read data from input paths
    X_test = pd.read_csv(X_test_input, sep=';')
    y_test = pd.read_csv(y_test_input, sep=';')
    model = joblib.load(model_input)
    run_id = pd.read_csv(run_id_input)
    run_id = run_id['run_id'][0]

    # Make predictions for the data
    y_test_pred = model.predict(X_test)
    y_test_pred_df = pd.DataFrame(y_test_pred, columns=['y_test_pred'])

    # Calculating test scores
    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    print(test_acc)
    print(test_f1)

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("test_f1_score", test_f1)

    # Save predictions to predictions path
    y_test_pred_df.to_csv(predictions_output, index=False, sep=';')
    