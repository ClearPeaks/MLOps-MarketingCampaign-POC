from pathlib import Path
from mldesigner import command_component, Input, Output

import pandas as pd

@command_component(
    name="feature_selection",
    version="1",
    display_name="Feature Selection",
    description="Selects the relevant features for the model.",
    environment=dict(
        conda_file=Path(__file__).parent / "conda.yaml",
        image="mcr.microsoft.com/azureml/curated/sklearn-1.1:17",
    ),
)

def feature_selection_component(
    input_data: Input(type='uri_file'),
    output_data: Output(type='uri_file'),
):

    # Read data from input path
    data = pd.read_csv(input_data, sep=';')

    # Variables that will be deleted from the dataset
    remove_list = ['ID', 'Year_Birth', 'NumDealsPurchases', 'NumStorePurchases',
                   'NumWebVisitsMonth', 'Complain', 'Z_CostContact', 'Z_Revenue']
    
    # Deleting the selected variables from the dataset
    data_selected = data.drop(remove_list, axis=1)

    # Saving to output path
    data_selected.to_csv(output_data, index=False, sep=';')