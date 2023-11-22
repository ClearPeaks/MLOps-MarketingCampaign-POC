from pathlib import Path
from mldesigner import command_component, Input, Output

import pandas as pd
from datetime import datetime

@command_component(
    name="feature_engineering",
    version="1",
    display_name="Feature Engineering",
    description="Transforms some features of the dataset.",
    environment=dict(
        conda_file=Path(__file__).parent / "conda.yaml",
        image="mcr.microsoft.com/azureml/curated/sklearn-1.1:17",
    ),
)

def feature_engineering_component(
    input_data: Input(type='uri_file'),
    output_data: Output(type='uri_file'),
):
    # Read data from input path
    data = pd.read_csv(input_data, sep=';')

    # Marital_Status =======================================
    # ======================================================
    
    # Merging the least frequent categories in Marital_Status
    data['Marital_Status'] = data['Marital_Status'].replace(['YOLO', 'Absurd', 'Alone'], 'Others')

    # Dt_Customer ==========================================
    # ======================================================

    # Convert to datetime
    date_objects = [datetime.strptime(date_str, "%Y-%m-%d") for date_str in data['Dt_Customer']]

    # The target date
    target_date = datetime(2023, 1, 1)

    # Calculate the difference in days between each date and the target date
    days_passed = [(target_date - date_object).days for date_object in date_objects]

    # Update the column
    data['Dt_Customer'] = days_passed

    # Saving to output path
    data.to_csv(output_data, index=False, sep=';')

    



