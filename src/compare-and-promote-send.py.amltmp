# Library imports
# ======================================================================================================
from azure.identity import DefaultAzureCredential           # Simplified way to obtain credentials
from azure.ai.ml import MLClient            # Interating with Azure ML services (datasets, moels, ...)
from azureml.core import Experiment
from azureml.core import Workspace
import time


# Get a handle to workspace
# ======================================================================================================
ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id="27a6aae6-ce60-4ae4-a06e-cfe9c1e824d4",
    resource_group_name="RG-ADA-MLOPS-POC",
    workspace_name="azu-ml-ada-mlops-poc",
)


# Wait for Azure ML pipeline to finish
# ======================================================================================================
ws = Workspace(subscription_id="27a6aae6-ce60-4ae4-a06e-cfe9c1e824d4",
               resource_group="RG-ADA-MLOPS-POC",
               workspace_name="azu-ml-ada-mlops-poc",)
experiment = Experiment(workspace=ws, name='marketing_pipeline_test_3')

status = ""
while status != "Completed":
    for run in experiment.get_runs():
        if run.get_status() == "Completed":
            status = "Completed"
        break
    time.sleep(1)


# Sumit job
# ======================================================================================================

# Configure job
job = command(
    code="./src",
    command="python compare-and-promote-script.py",
    environment="mcr.microsoft.com/azureml/curated/sklearn-1.1:17",
    compute="shared-compute-poc",
    display_name="Compare and promote model",
    experiment_name="marketing_pipeline_test_3"
)

# Submit job
ml_client.create_or_update(job)