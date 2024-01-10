# Library imports
# ======================================================================================================
import mlflow
from azureml.core import Experiment
from azureml.core import Workspace
import time


# Wait for Azure ML pipeline to finish
# ======================================================================================================
ws = Workspace(subscription_id="27a6aae6-ce60-4ae4-a06e-cfe9c1e824d4",
               resource_group="RG-ADA-MLOPS-POC",
               workspace_name="azu-ml-ada-mlops-poc",)
experiment = Experiment(workspace=ws, name='marketing-pipeline-demo-v3')

status = ""
while status != "Completed":
    for run in experiment.get_runs():
        if run.display_name == 'marketing_campaign_prediction' and run.get_status() == "Running":
            break
        status = 'Completed'
    time.sleep(2)


# Coompare and promote model
# ======================================================================================================

# Setting the mlflow client for advanced operations
client = mlflow.tracking.MlflowClient()

model_name = "Response-Predictor"

production_models = client.get_latest_versions(model_name, stages=["Production"])
last_model = client.get_latest_versions(model_name, stages=["None"])[0]

# No models in prdocution yet
if len(production_models) == 0:
    client.transition_model_version_stage(model_name, last_model.version, "Production")

# Compare with model in production
else:
    # Obtaining production model metrics
    prod_run_id = production_models[0].run_id
    prod_run = client.get_run(prod_run_id)
    prod_metrics = prod_run.data.metrics
    prod_accuracy = prod_metrics['test_accuracy']
    prod_f1 = prod_metrics['test_f1_score']

    # Obtaining this run's model metrics
    new_metrics = client.get_run(last_model.run_id).data.metrics
    new_accuracy = new_metrics['test_accuracy']
    new_f1 = new_metrics['test_f1_score']

    # Promoting new model if it is better than production model
    if new_accuracy > prod_accuracy and new_f1 > prod_f1:
        # obtain model versions            
        to_prod_version = last_model.version
        to_none_version = client.search_model_versions("run_id='{}'".format(prod_run_id))[0].version

        # Transition new model to Production stage and old model to None
        client.transition_model_version_stage(model_name, to_prod_version, "Production")
        client.transition_model_version_stage(model_name, to_none_version, "None")