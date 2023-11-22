from pathlib import Path
from mldesigner import command_component, Input, Output

import pandas as pd
import numpy as np

@command_component(
    name="outlier_treatment",
    version="1",
    display_name="Outlier Treatment",
    description="Deals with the outliers in the data.",
    environment=dict(
        conda_file=Path(__file__).parent / "conda.yaml",
        image="mcr.microsoft.com/azureml/curated/sklearn-1.1:17",
    ),
)

def outlier_treatment_component(
    input_data: Input(type='uri_file'),
    output_data: Output(type='uri_file'),
):
    # Read data from input path
    data = pd.read_csv(input_data, sep=';')

    # Determining which are outliers
    condition = data['Income'] == 666666
    
    # Setting outlier to NaN
    data.loc[condition, 'Income'] = np.nan

    # Saving to output path
    data.to_csv(output_data, index=False, sep=';')

    