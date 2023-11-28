# Library imports
# ======================================================================================================
from azure.identity import DefaultAzureCredential           # Simplified way to obtain credentials

from azure.ai.ml import MLClient            # Interating with Azure ML services (datasets, moels, ...)
from azure.ai.ml.dsl import pipeline        # Define machine learning pipelines
from azure.ai.ml import load_component      # Load Azure ML components

from azure.ai.ml import Input                       # Specify inputs to ML jobs
from azure.ai.ml.constants import AssetTypes        # Provide standarized identifiers for assets


# Get a handle to workspace
# ======================================================================================================
ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id="27a6aae6-ce60-4ae4-a06e-cfe9c1e824d4",
    resource_group_name="RG-ADA-MLOPS-POC",
    workspace_name="azu-ml-ada-mlops-poc",
)

# Define te compute that is going to be used to run the pipeline
cpu_compute_target = "default-compute-poc"


# Prepare data
# ======================================================================================================
available_data = Input(
    type=AssetTypes.URI_FILE,
    path="azureml:data_available:2"
)


# Load components
# ======================================================================================================
components = []

from feature_selection.feature_selection_component import feature_selection_component
components.append(["feature_selection", feature_selection_component])

from feature_engineering.feature_engineering_component import feature_engineering_component
components.append(['feature_engineering', feature_engineering_component])

from outlier_treatment.outlier_treatment_component import outlier_treatment_component
components.append(['outlier_treatment', outlier_treatment_component])

from split_data.split_data_component import split_data_component
components.append(['split_data', split_data_component])

from imputation.imputation_component import imputation_component
components.append(['imputation', imputation_component])

from normalization.normalization_component import normalization_component
components.append(['normalization', normalization_component])

from encoding.encoding_component import encoding_component
components.append(['encoding', encoding_component])

from training.training_component import training_component
components.append(['training', training_component])

from scoring.scoring_component import scoring_component
components.append(['scoring', scoring_component])


# Build pipeline
# ======================================================================================================
@pipeline(
    default_compute=cpu_compute_target,
)

def marketing_campaign_prediction(pipeline_input_data):
    
    # Feature selection
    feature_selection_node = feature_selection_component(input_data=pipeline_input_data)

    # Feature engineering
    feature_engineering_node = feature_engineering_component(input_data=feature_selection_node.outputs.output_data)
    
    # Outlier treatment
    outlier_treatment_node = outlier_treatment_component(input_data=feature_engineering_node.outputs.output_data)
    
    # Split data
    split_data_node = split_data_component(input_data=outlier_treatment_node.outputs.output_data)
    
    # Imputation
    imputation_node = imputation_component(X_train_input=split_data_node.outputs.X_train_data,
                                           X_test_input=split_data_node.outputs.X_test_data)
    # Normalization
    normalization_node = normalization_component(X_train_input=imputation_node.outputs.X_train_output,
                                                 X_test_input=imputation_node.outputs.X_test_output)
    # Encoding
    encoding_node = encoding_component(X_train_input=normalization_node.outputs.X_train_output,
                                       X_test_input=normalization_node.outputs.X_test_output)
    # Training
    training_node = training_component(X_train_input=encoding_node.outputs.X_train_output,
                                       y_train_input=split_data_node.outputs.y_train_data)
    # Scoring
    scoring_node = scoring_component(X_test_input=encoding_node.outputs.X_test_output,
                                     y_test_input=split_data_node.outputs.y_test_data,
                                     model_input=training_node.outputs.model_output)

# Instantiate the pipeline
pipeline_job = marketing_campaign_prediction(pipeline_input_data=available_data)


# Submit pipeline
# ======================================================================================================
pipeline_job = ml_client.jobs.create_or_update(
    pipeline_job, experiment_name="marketing_pipeline_test_3"
)


# Register components
# ======================================================================================================
for component in components:
    # try:
    #     # try get back the component
    #     retrieved_component = ml_client.components.get(name=component[0], version="1")
    # except:
    #     # if not exists, register component using following code
    retrieved_component = ml_client.components.create_or_update(component[1])