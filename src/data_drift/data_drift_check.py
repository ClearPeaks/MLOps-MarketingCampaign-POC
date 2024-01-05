#!/usr/bin/env python
# coding: utf-8

# ### Library imports

# COMPONENT
from pathlib import Path
from mldesigner import command_component, Input, Output

# AZURE
from azure.identity import DefaultAzureCredential          
from azure.ai.ml import MLClient                           
from azureml.core import Workspace, Datastore, Dataset
from azure.ai.ml import command
from azure.ai.ml.entities import Environment

# DATADRIFT
from azureml.datadrift import DataDriftDetector
from azureml.datadrift import AlertConfiguration

# OTHERS
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split


@command_component(
    name="data_drift_detector_component",
    display_name="Data drift detection",
    description="Checks for data drift in new data",
    environment=dict(
        conda_file=Path(__file__).parent / "conda-ddd.yaml",
        image="mcr.microsoft.com/azureml/curated/sklearn-1.1:17",
    ),
    is_deterministic = False
)

def data_drift_component():
    # ### 1. Get handle to workspace + other useful definitions

    # Setting the ML client to perform advanced operations
    ml_client = MLClient(
        DefaultAzureCredential(),
        subscription_id="27a6aae6-ce60-4ae4-a06e-cfe9c1e824d4",
        resource_group_name="RG-ADA-MLOPS-POC",
        workspace_name="azu-ml-ada-mlops-poc",
    )

    # Getting a workspace object
    ws = Workspace(subscription_id="27a6aae6-ce60-4ae4-a06e-cfe9c1e824d4",
                resource_group="RG-ADA-MLOPS-POC",
                workspace_name="azu-ml-ada-mlops-poc",)

    # Getting the default datastore of the workspace
    dstore = ws.get_default_datastore()

    # The names of the different datasets used
    available_data_name = 'data_available_t'
    simulation_data_name = 'data_simulation'
    baseline_data_name = 'baseline'
    target_data_name = 'target'
    waiting_data_name = 'waiting_available'

    # Other names
    data_drift_detector_name = 'drift_monitor'  # Name of the already existing or to-create data drift detector
    date_column_name = 'date'                   # Column corresponding to the timestamp to treat dsets as time series
    compute_drift_checks = 'shared-compute-poc' # Compute that will perform the data drift checks
    check_frequency = 'Day'                     # How the data drift detector should analyse the data (daily, weekly, ...)
    alert_emails = ['alejandro.donaire@clearpeaks.com', # Send email to these if data drift
                    'alex.romero@clearpeaks.com'
                ]   
    features_check = ['Year_Birth', 'Education', 'Marital_Status', 'Income', 'Kidhome',    # List of features
                    'Teenhome', 'Dt_Customer', 'Recency', 'MntWines', 'MntFruits',         # we want to check data drift on.
                    'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts',              # Should be less than 200
                    'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases',
                    'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth',
                    'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1',
                    'AcceptedCmp2', 'Complain', 'Z_CostContact', 'Z_Revenue', 'Response']

    # Numeric constants
    drift_threshold = 0.6
    num_new_samples = 50   # Number of new rows every time we get new data
    rows_per_day = 50      # Number of rows per day
                        # If the data drift detection is performed every day
                        # the constants above should be equal



    # ### 2. Retrieve new data
    # The following is just a simulation of receiving new unseen data.

    # Function to generate synthetic data partially based on dataset characterisics
    def generate_new_data(dataframe, num_samples):
        new_data = {}
        for column_name in dataframe.columns:
            if column_name in ['ID', 'Year_Birth', 'Education', 'Marital_Status', 'Kidhome', 'Teenhome',
                            'Dt_Customer', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1',
                            'AcceptedCmp2', 'Complain', 'Z_CostContact', 'Z_Revenue', 'Response']:
                # For categorical columns: pandas' sample method
                new_data[column_name] = dataframe[column_name].sample(n=num_samples, replace=True).reset_index(drop=True)
            else:
                # For non-categorical columns: numpy's random.normal to sample from a normal distribution
                mean_value = dataframe[column_name].mean()
                std_dev_value = dataframe[column_name].std()
                new_data[column_name] = np.abs(np.random.normal(loc=mean_value, scale=std_dev_value, size=num_samples))
        
        return pd.DataFrame(new_data)

    # Get the latest version of the simulation data
    data_simulation_versions = []
    for data_simulation_version in ml_client.data.list(name=simulation_data_name):
        data_simulation_versions.append(data_simulation_version.version)

    # Retrieve the simulation data asset
    data_simulation = ml_client.data.get(name=simulation_data_name, version=max(data_simulation_versions))
    data_simulation_df = pd.read_csv(data_simulation.path, sep=";")

    # Generate new synthetic data
    new_data = generate_new_data(data_simulation_df, num_new_samples)



    # ### 3. Retrieve data drift monitor and update target dataset

    # Retrieving the names of the already existing data drift detectors
    existing_data_drift_detectors = [DDD.name for DDD in DataDriftDetector.list(ws)]

    # Check if the data drift detector exists
    if data_drift_detector_name in existing_data_drift_detectors:
        # If data detector exists, target dataset too

        # UPDATE TARGET DATASET
        # =====================

        # Retrieve target dataset
        target = Dataset.get_by_name(workspace=ws, name=target_data_name)
        target = target.to_pandas_dataframe()

        # Get the last date of the target dataset
        last_date = max(target[date_column_name])

        # Define the starting data for the new chunk of data
        start_date = last_date + timedelta(days=1)

        # Add the data column to the new data
        new = new_data.copy()
        new[date_column_name] = [start_date + timedelta(days=i//rows_per_day) for i in range(len(new))]

        # Merge target dataset with new data
        new_target = pd.concat([target, new_data], axis=0)

        # Update target dataset (register new version)
        ds = Dataset.Tabular.register_pandas_dataframe(
                dataframe = new_target,
                name = target_data_name,
                description = "Target dataset to test data drift",
                target = dstore
            )

        # Assign the timestamp attribute
        target = Dataset.get_by_name(workspace=ws, name=target_data_name)
        target = target.with_timestamp_columns(date_column_name)
        target.register(ws, target_data_name, create_new_version=True)
        
        # CREATING THE DATA DRIFT MONITOR
        # ===============================

        # NOTE: This step might seem might seem redundant or wrong, but,
        # given the current capabilities of the DataDrfitDetector class as of
        # january 2024, the target dataset can't be updated (except for a special
        # case when using an AKS cluster), and so a new data drift monitor
        # must be created with the updated target dataset.

        # Delete old monitor
        monitor = DataDriftDetector.get_by_name(ws, data_drift_detector_name)
        monitor.delete(wait_for_completion=True)

        # Retrieve baseline and target datasets
        baseline = Dataset.get_by_name(workspace=ws, name=baseline_data_name)
        target = Dataset.get_by_name(workspace=ws, name=target_data_name)

        # Specify e-mail adress to send alert for data drift
        alert_config = AlertConfiguration(alert_emails)

        # Set up data drift detector
        monitor = DataDriftDetector.create_from_datasets(ws, data_drift_detector_name, baseline, target,
                                                            compute_target=compute_drift_checks,
                                                            frequency=check_frequency,
                                                            feature_list=features_check,
                                                            drift_threshold=drift_threshold,
                                                            alert_config=alert_config
                                                            )

    else:
        # The data drift detector does not exist so it should be 
        # created along with the target and baseline datasets
        
        # CREATING BASELINE DATASET
        # =========================

        # Retrieve available data, get latest version
        data_available = Dataset.get_by_name(ws, name=available_data_name)
        data_available = data_available.to_pandas_dataframe()

        # Splitting the data like in the split component
        X = data_available.drop(['Response'], axis=1)
        y = data_available['Response']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=data_available['Response'])

        # Recreating the training data with the target
        baseline = pd.concat([X_train, y_train], axis=1)

        # Start date (random)
        start_date = datetime(2023, 1, 1)

        # Adding a date column: 1 day passes every 40 rows
        baseline[date_column_name] = [start_date + timedelta(days=i//(int(rows_per_day*0.8))) for i in range(len(baseline))]

        # Register dataframe as dataset (if it already exists, it updates the version)
        ds = Dataset.Tabular.register_pandas_dataframe(
                dataframe = baseline,
                name = baseline_data_name,
                description = "Baseline dataset to test data drift",
                target = dstore
            )
        
        # Assign the timestamp attribute
        baseline = Dataset.get_by_name(workspace=ws, name=baseline_data_name)
        baseline = baseline.with_timestamp_columns(date_column_name)
        baseline.register(ws, baseline_data_name, create_new_version=True)

        # CREATING TARGET DATASET
        # =======================

        # Retrieving the last date from the baseline dataset
        baseline = baseline.to_pandas_dataframe()
        last_date = max(baseline[date_column_name])

        # The start date for the new data is the last date of the baseline plus one day
        start_date = last_date + timedelta(days=1)

        # Creating a copy of the new data
        new_target = new_data.copy()

        # Adding a date column: 1 day passes every 50 rows
        new_target[date_column_name] = [start_date + timedelta(days=i//rows_per_day) for i in range(len(new_target))]

        # Register dataframe as dataset (if it already exists, it updates the version)
        ds = Dataset.Tabular.register_pandas_dataframe(
                dataframe = new_target,
                name = target_data_name,
                description = "Target dataset to test data drift",
                target = dstore
            )
        
        # Assign the timestamp attribute
        target = Dataset.get_by_name(workspace=ws, name=target_data_name)
        target = target.with_timestamp_columns(date_column_name)
        target.register(ws, target_data_name, create_new_version=True)

        # CREATING THE DATA DRIFT MONITOR
        # ===============================

        # Retrieve baseline and target datasets
        baseline = Dataset.get_by_name(workspace=ws, name=baseline_data_name)
        target = Dataset.get_by_name(workspace=ws, name=target_data_name)

        # Specify e-mail adress to send alert for data drift
        alert_config = AlertConfiguration(alert_emails)

        # Set up data drift detector
        monitor = DataDriftDetector.create_from_datasets(ws, data_drift_detector_name, baseline, target,
                                                            compute_target=compute_drift_checks,
                                                            frequency=check_frequency,
                                                            feature_list=features_check,
                                                            drift_threshold=drift_threshold,
                                                            alert_config=alert_config
                                                            )
        
        # CREATING THE WAITING AVAILABLE DATASET
        # ======================================

        # New dataframe with no columns 
        waiting_availabe_blueprint = pd.DataFrame(columns=new_data.columns)
        # Adding a new row with the same string in all columns.
        # Adding at least one row is necessary to resgister the dataset later
        waiting_availabe_blueprint.loc[len(waiting_availabe_blueprint)] = 'No Data'

        ds = Dataset.Tabular.register_pandas_dataframe(
            dataframe = waiting_availabe_blueprint,
            name = waiting_data_name,
            description = '''This dataset serves as a sort of data warehouse to store new data 
                        when data drift is not detected. This data will be concatenated with
                        the available data once data drift is detected so a new model can be trained''',
            target = dstore
        )



    # ### 4. Run data drift detection job

    # Retrieve updated target dataset
    target = Dataset.get_by_name(workspace=ws, name=target_data_name)
    target = target.to_pandas_dataframe()

    # Start and end of the data drift detection
    # Only performs detection on new data
    end_backfill = max(target[date_column_name])
    start_backfill = end_backfill - timedelta(days=int(num_new_samples/rows_per_day))

    # Start data drift detection job
    backfill = monitor.backfill(start_backfill, end_backfill).wait_for_completion()



    # ### 5. Retrieve data drift detection metrics

    # Retrieve everything from the run
    metrics = monitor.get_output(start_backfill, end_backfill)
    drift_coefficient = metrics[1][0]['metrics'][0]['dataset_metrics'][0]['value']



    # ### 6. Update datasets depending on whether there is data drift or not

    # Check if there is drift in at least one day
    if drift_coefficient > drift_threshold:
        # There is drift in the new data
        
        # AVAILABLE DATA UPDATE (train + test)
        # ====================================

        # Retrieve the available data asset
        data_available = Dataset.get_by_name(ws, name=available_data_name)
        data_available = data_available.to_pandas_dataframe()

        # Retrieve last version of the waiting data
        waiting_data = Dataset.get_by_name(ws, name=waiting_data_name)
        waiting_data = waiting_data.to_pandas_dataframe()

        # Check if waiting data is empty or not
        if all(waiting_data.iloc[-1] == 'No Data'):
            # Waiting data is empty
            # Concatenating the available data with the new data
            new_available = pd.concat([data_available, new_data], axis=0)
        else:
            # Waiting data contains data
            # Concatenating the available data with the waiting data and the new data
            new_available = pd.concat([data_available, waiting_data, new_data], axis=0)

        # Update available dataset (register new version)
        ds = Dataset.Tabular.register_pandas_dataframe(
                dataframe = new_available,
                name = available_data_name,
                target = dstore, 
            )

        # UPDATE BASELINE DATA (train)
        # ============================

        # Splitting the data like in the split component
        X = new_available.drop(['Response'], axis=1)
        y = new_available['Response']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=new_available['Response'])

        # Recreating the training data with the target
        new_baseline = pd.concat([X_train, y_train], axis=1)

        # Adding the timestamp
        new_baseline[date_column_name] = [start_date + timedelta(days=i//(int(rows_per_day*0.8))) for i in range(len(new_baseline))]

        ds = Dataset.Tabular.register_pandas_dataframe(
                dataframe = new_baseline,
                name = baseline_data_name,
                description = "Baseline dataset to test data drift",
                target = dstore
            )

        # Assign the timestamp attribute
        baseline = Dataset.get_by_name(workspace=ws, name=baseline_data_name)
        baseline = baseline.with_timestamp_columns(date_column_name)
        baseline.register(ws, baseline_data_name, create_new_version=True)
        
        # RESET WAITING DATA
        # ==================
        
        # New dataframe with no columns 
        waiting_availabe_blueprint = pd.DataFrame(columns=new_available.columns)
        # Adding a new row with the same string in all columns.
        # Adding at least one row is necessary to resgister the dataset later
        waiting_availabe_blueprint.loc[len(waiting_availabe_blueprint)] = 'No Data'

        ds = Dataset.Tabular.register_pandas_dataframe(
            dataframe = waiting_availabe_blueprint,
            name = waiting_data_name,
            description = '''This dataset serves as a sort of data warehouse to store new data 
                        when data drift is not detected. This data will be concatenated with
                        the available data once data drift is detected so a new model can be trained''',
            target = dstore
        )

        # RETRAIN MODEL WITH NEW AVAILABLE DATA
        # =====================================

        print('Data drift detected. Retraining model.')
        
        # Send job to retrain model, executing the Azure ML Pipeline
        create_and_run_pipeline_job = command(
            code="../",
            command="python create-and-run-pipeline.py",
            environment=Environment(
                conda_file="conda-ddd.yaml",
                image="mcr.microsoft.com/azureml/curated/mldesigner-minimal:18"),
            compute="shared-compute-poc",
            display_name="Retrain model",
            experiment_name="marketing-pipeline-demo-v2"
        )

        # Submit job
        ml_client.create_or_update(create_and_run_pipeline_job)

    else:
        # There isn't drift

        # WAITING DATA UPDATE (train + test)
        # ====================================

        # Retrieve last version of the waiting data
        waiting_data = Dataset.get_by_name(ws, name=waiting_data_name)
        waiting_data = waiting_data.to_pandas_dataframe()

        # Check if waiting data is empty or not
        if all(waiting_data.iloc[-1] == 'No Data'):
            # Waiting data is empty
            # New waiting data is simply the new data of this run
            new_waiting = new_data.copy()
        else:
            # Waiting data contains data
            # Concatenating the waiting data with the new data
            new_waiting = pd.concat([waiting_data, new_data], axis=0)

        ds = Dataset.Tabular.register_pandas_dataframe(
            dataframe = new_waiting,
            name = waiting_data_name,
            description = '''This dataset serves as a sort of data warehouse to store new data 
                        when data drift is not detected. This data will be concatenated with
                        the available data once data drift is detected so a new model can be trained''',
            target = dstore
        )


