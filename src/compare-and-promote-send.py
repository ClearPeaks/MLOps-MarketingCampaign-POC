# Library imports
# ======================================================================================================
from azure.identity import DefaultAzureCredential           # Simplified way to obtain credentials
from azureml.core.authentication import MsiAuthentication
from azure.ai.ml import MLClient            # Interating with Azure ML services (datasets, moels, ...)
from azure.ai.ml import command
from azureml.core import Experiment
from azureml.core import Workspace
import time



# Get a handle to workspace
# ======================================================================================================
ml_client = MLClient(
    MsiAuthentication(),
    subscription_id="27a6aae6-ce60-4ae4-a06e-cfe9c1e824d4",
    resource_group_name="RG-ADA-MLOPS-POC",
    workspace_name="azu-ml-ada-mlops-poc",
)


# Wait for Azure ML pipeline to finish
# ======================================================================================================
ws = Workspace(subscription_id="27a6aae6-ce60-4ae4-a06e-cfe9c1e824d4",
               resource_group="RG-ADA-MLOPS-POC",
               workspace_name="azu-ml-ada-mlops-poc",)
experiment = Experiment(workspace=ws, name='marketing-pipeline-demo-v2')

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
    environment="AzureML-sklearn-0.24-ubuntu18.04-py37-cpu@latest",
    compute="shared-compute-poc",
    display_name="Compare and promote model",
    experiment_name="marketing-pipeline-demo-v2"
)

# Submit job
ml_client.create_or_update(job)