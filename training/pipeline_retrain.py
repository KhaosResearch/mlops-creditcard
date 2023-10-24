
import json
import os
import yaml
import tempfile
from io import BytesIO
from os.path import join

import joblib
import numpy as np
import pandas as pd
import mlflow
import boto3
from kubernetes import client, config
from sklearn.ensemble import RandomForestClassifier
from prefect import flow, task
from prefect.blocks.kubernetes import KubernetesClusterConfig
from creditcard_mlflow_wrapper import CreditcardMlflowWrapper

@flow
def retraining_flow_creditcard(data_bucket: str="mlops-am", data_folder: str="data/creditcard", model_name: str="creditcard", test_size: float=0.1):
    new_data = ingest_new_data(data_bucket, data_folder)
    retrain_model(new_data, model_name, test_size)
    deploy_best_model(model_name, wait_for=[retrain_model])

@task
def deploy_best_model(model_name):

    mlflow_client = mlflow.client.MlflowClient()

    model_versions_metadata = mlflow_client.search_model_versions(f"name='{model_name}'").to_list()

    production_run_id = next(filter(lambda x: x.current_stage == "Production", model_versions_metadata)).run_id
    production_accuracy = mlflow_client.get_metric_history(production_run_id, key="accuracy")[0].value


    latest_run_id = max(model_versions_metadata, key=lambda x: int(x.version)).run_id
    latest_version = max(model_versions_metadata, key=lambda x: int(x.version)).version
    latest_accuracy = mlflow_client.get_metric_history(latest_run_id, key="accuracy")[0].value

    #is_new_model_better =  latest_accuracy > production_accuracy
    is_new_model_better =  True

    if is_new_model_better:

        print("deploying retrained model")

        cluster_config_block = KubernetesClusterConfig.load("k8s-config")
        config.load_kube_config_from_dict(cluster_config_block.config)
        custom_api = client.CustomObjectsApi()
        with open("files/deploy_creditcard.yaml", 'r') as stream:
            deployment_yaml = yaml.safe_load(stream)

        # If value is empty in deployment file and defined in environ, replace with environment value
        for i, env in enumerate(deployment_yaml["spec"]["predictors"][0]["componentSpecs"][0]["spec"]["containers"][0]["env"]):
            if env["name"] in os.environ and env["value"] == "":
                deployment_yaml["spec"]["predictors"][0]["componentSpecs"][0]["spec"]["containers"][0]["env"][i] = os.environ[env["name"]]

        try:
            mlflow_client.transition_model_version_stage(model_name, latest_version, stage="Production", archive_existing_versions=True)

            custom_api.create_namespaced_custom_object(
                group="machinelearning.seldon.io",
                version="v1alpha2",
                namespace="mlops-seldon",
                plural="seldondeployments",
                body=deployment_yaml
            )
        except:
            existing_deployment_yaml = custom_api.get_namespaced_custom_object(
                group="machinelearning.seldon.io",
                version="v1alpha2",
                namespace="mlops-seldon",
                plural="seldondeployments",
                name=deployment_yaml["metadata"]["name"],
            )

            custom_api.delete_namespaced_custom_object(
                group="machinelearning.seldon.io",
                version="v1alpha2",
                namespace="mlops-seldon",
                plural="seldondeployments",
                name=existing_deployment_yaml["metadata"]["name"],
            )

            custom_api.create_namespaced_custom_object(
                group="machinelearning.seldon.io",
                version="v1alpha2",
                namespace="mlops-seldon",
                plural="seldondeployments",
                body=deployment_yaml
            )

@task
def retrain_model(new_data: pd.DataFrame, model_name: str, test_size: float, n_jobs: int=2):
    model_uri = f"models:/{model_name}/production"
    download_path = mlflow.artifacts.download_artifacts(artifact_uri=model_uri)
    production_testing_dataset = pd.read_csv(os.path.join(download_path, "artifacts", "testing_dataset.csv"))
    production_training_dataset = pd.read_csv(os.path.join(download_path, "artifacts", "training_dataset.csv"))

    new_data = new_data.replace([np.inf, -np.inf], np.nan)
    new_data = new_data.fillna(np.nan)
    new_data = new_data.dropna()

    test_df = new_data[:int(test_size*len(new_data))]
    train_df = new_data[int(test_size*len(new_data)):]

    test_df = pd.concat([test_df, production_testing_dataset], axis=0)
    train_df = pd.concat([train_df, production_training_dataset], axis=0)

    X_test = test_df.drop(["Class"], axis=1)
    X_train = train_df.drop(["Class"], axis=1)
    y_test = test_df["Class"]
    y_train = train_df["Class"]

    # Train model
    clf = RandomForestClassifier(n_jobs=n_jobs)
    clf.fit(X_train, y_train)
    y_true = clf.predict(X_test)

    accuracy = np.mean(y_true == y_test)

    tempdir = tempfile.gettempdir()

    model_name = "model.joblib"
    model_path = join(tempdir, model_name)
    joblib.dump(clf, model_path)

    training_data_name = "training_dataset.csv"
    training_data_path = join(tempdir, training_data_name)
    train_df.to_csv(training_data_path, index=False)

    testing_data_name = "testing_dataset.csv"
    testing_data_path = join(tempdir, testing_data_name)
    test_df.to_csv(testing_data_path, index=False)

    model_metadata = {
        "model": str(type(clf)),
        "n_jobs": str(clf.n_jobs),
        "used_columns": [str(column) for column in X_train.columns],
        "classes": [str(class_) for class_ in clf.classes_],
        "test_size": str(test_size)
    }

    model_metadata_name = "metadata.json"
    model_metadata_path = join(tempdir, model_metadata_name)

    with open(model_metadata_path, "w") as f:
        json.dump(model_metadata, f)

    experiment_id = mlflow.get_experiment_by_name("creditcard").experiment_id

    with mlflow.start_run(experiment_id=experiment_id):

        mlflow.pyfunc.log_model(artifact_path="model",
                                registered_model_name="creditcard",
                                code_path=["creditcard_mlflow_wrapper.py"],
                                pip_requirements=["mlflow==2.3.1", "joblib==1.1.1", "creditcard>=1.0.0"],
                                python_model=CreditcardMlflowWrapper(),
                                artifacts={
                                    "model_file": model_path, 
                                    "training_data": training_data_path, 
                                    "testing_data": testing_data_path, 
                                    "metadata_file": model_metadata_path,
                                },
        )
        
        mlflow.log_params(model_metadata)

        mlflow.log_metric("accuracy", accuracy)    

@task
def ingest_new_data(bucket: str, folder: str):

    endpoint_url = os.environ.get('MLFLOW_S3_ENDPOINT_URL')
    aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY')

    s3 = boto3.client('s3', endpoint_url=endpoint_url, aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)

    response = s3.list_objects_v2(Bucket=bucket, Prefix=folder)
    csv_files = [obj['Key'] for obj in response.get('Contents', []) if obj['Key'].endswith('.csv')]

    dfs = []

    for csv_file in csv_files:
        response = s3.get_object(Bucket=bucket, Key=csv_file)
        csv_content = response['Body'].read()
        csv_buffer = BytesIO(csv_content)
        df = pd.read_csv(csv_buffer, index_col='id')
        dfs.append(df)

    merged_df = pd.concat(dfs, ignore_index=True)
    return merged_df