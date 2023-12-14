# Library imports
# ======================================================================================================
# AZURE
from azure.identity import DefaultAzureCredential           # Simplified way to obtain credentials for Azure services
from azure.ai.ml import MLClient                            # Client for interacting with Azure Machine Learning services, including managing datasets, models, etc.
from azure.ai.ml.entities import Environment                # Represents an environment in Azure ML, used to define runtime context
from azure.ai.ml.entities import ManagedOnlineDeployment    # Used to configure and manage online deployments in Azure ML
from azure.ai.ml.entities import ManagedOnlineEndpoint    # Used to configure and manage online deployments in Azure ML
from azure.ai.ml.entities import CodeConfiguration          # Specifies code configuration for Azure ML projects, including script and scoring file details

# MLFLOW
import mlflow                                               # Managing the end-to-end machine learning lifecycle
from mlflow.deployments import get_deploy_client            # Function to obtain a client for deploying ML models via MLflow


# Get handle to workspace
# ======================================================================================================
ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id="27a6aae6-ce60-4ae4-a06e-cfe9c1e824d4",
    resource_group_name="RG-ADA-MLOPS-POC",
    workspace_name="azu-ml-ada-mlops-poc",
)


# Define the clients
# ======================================================================================================
# MLFlow client
client = mlflow.tracking.MlflowClient()

# MLFlow deployment client
deployment_client = get_deploy_client(mlflow.get_tracking_uri())


# Define the names
# ======================================================================================================
# Model name
model_name = "Marketing-Response-Predictor"

# Endpoint name
endpoint_name = "MC-Prod-Endpoint-4242"


# Start with the logic of the deployment
# ======================================================================================================

# Get version of the last model trained
latest_version = client.search_model_versions(f"name='{model_name}'")[0].version

# Get version of the model in production
prod_model = client.get_latest_versions(model_name, stages=["Production"])[0]
prod_version = prod_model.version

# Check if the versions are the same
if latest_version == prod_version:
    print("The new model should be deployed")

    # Define the model that is going to be deployed
    # =================================================================
    model_name = prod_model.name + ":" + prod_version
    print("The model going to be deployed is:", model_name)


    # Get the endpoint
    # ================================================================
    # If endpoint doesn't exist, create it
    try:
        endpoint = ml_client.online_endpoints.get(name=endpoint_name)
    except:
        endpoint = ManagedOnlineEndpoint(
            name = endpoint_name, 
            description="Inference endpoint",
            auth_mode="key"
        )
        ml_client.online_endpoints.begin_create_or_update(endpoint).result()
    

    # Define the environment of the new deployment
    # ================================================================
    environment = Environment(
        conda_file="conda.yaml",
        image="mcr.microsoft.com/azureml/curated/minimal-ubuntu20.04-py38-cpu-inference:21",
    )


    # Instantiate the new deployment
    # ===============================================================
    # Defining the name of the deployment with the name of the new model
    deployment_name = "dep-" + prod_model.name.lower() + "-" + prod_version

    # Instantiating the deployment with the desired configuration
    deployment = ManagedOnlineDeployment(
        name=deployment_name,
        endpoint_name=endpoint_name,
        model=model_name,
        environment=environment,
        code_configuration=CodeConfiguration(
            code=".",
            scoring_script="score.py"
        ),
        instance_type="Standard_D2as_v4",
        instance_count=1,
    )
    print("The name of the new deployment is:", deployment_name)


    # Create the new deployment
    # ===============================================================
    deployment = ml_client.online_deployments.begin_create_or_update(
        deployment
    ).result()


    # Check if there isn't any deployment yet or if there is one already
    # ===============================================================
    # Get all deployments in the endpoint (iterator)
    deployments = ml_client.online_deployments.list(endpoint_name=endpoint_name)

    # Make a list of the deployments
    dep_list = []
    for deployment in deployments:
        dep_list.append(deployment)

    # Check if there is or isn't a deployment already
    if len(dep_list) == 1:
        print("There isn't any deployment yet")

        # Send all traffic to the new deployment
        endpoint = ml_client.online_endpoints.get(endpoint_name)
        endpoint.traffic = {deployment_name: 100}
        ml_client.online_endpoints.begin_create_or_update(endpoint).result()

    else:
        print("There is one deployment already")

        # Switch the entire traffic to the new deployment
        endpoint = ml_client.online_endpoints.get(endpoint_name)
        endpoint.traffic = {dep_list[1].name: 0, deployment_name: 100}
        ml_client.online_endpoints.begin_create_or_update(endpoint).result()

        # Delete old deployment
        ml_client.online_deployments.begin_delete(
            name=dep_list[1].name, 
            endpoint_name=endpoint_name
        )
    
else:
    print("The new model should NOT be deployed")


