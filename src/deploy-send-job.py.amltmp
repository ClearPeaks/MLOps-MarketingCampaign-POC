# Library imports
# ======================================================================================================
from azure.identity import DefaultAzureCredential           # Simplified way to obtain credentials
from azure.ai.ml import MLClient            # Interating with Azure ML services (datasets, moels, ...)
from azure.ai.ml import command
from azure.ai.ml.entities import Environment                # Represents an environment in Azure ML, used to define runtime context


# Get a handle to workspace
# ======================================================================================================
ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id="27a6aae6-ce60-4ae4-a06e-cfe9c1e824d4",
    resource_group_name="RG-ADA-MLOPS-POC",
    workspace_name="azu-ml-ada-mlops-poc",
)


# Sumit job
# ======================================================================================================

# Configure job
job = command(
    code="./src",
    command="python deploy-script.py",
    environment=Environment(
        conda_file="./src/conda-deploy-job.yaml",
        image="mcr.microsoft.com/azureml/curated/minimal-ubuntu20.04-py38-cpu-inference:21",
    ),
    compute="shared-compute-poc",
    display_name="Deploy",
    experiment_name="marketing-pipeline-demo-v3"
)

# Submit job
ml_client.create_or_update(job)