import mlflow
import json
import os
import numpy as np
import pandas as pd
from prefect.filesystems import S3

from creditcard_mlflow_wrapper import CreditcardMlflowWrapper

s3_block = S3.load("khaos-minio")

os.environ["AWS_ACCESS_KEY_ID"] = s3_block.aws_access_key_id.get_secret_value()
os.environ["AWS_SECRET_ACCESS_KEY"] = s3_block.aws_secret_access_key.get_secret_value()
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://192.168.219.2:9000"
os.environ["MLFLOW_S3_IGNORE_TLS"] = "true"

model_file = "base_model/model.joblib"
metadata_file = "base_model/metadata.json"
training_data = "base_model/training_dataset.csv"
testing_data = "base_model/testing_dataset.csv"

mlflow.set_tracking_uri("http://192.168.219.71:32001")

experiment_id = mlflow.get_experiment_by_name("creditcard").experiment_id

with mlflow.start_run(experiment_id=experiment_id):
    mlflow.pyfunc.log_model(artifact_path="model",
                            registered_model_name="creditcard",
                            code_path=["creditcard_mlflow_wrapper.py"],
                            pip_requirements=["mlflow==2.3.1", "joblib==1.1.1", "creditcard>=1.0.0"],
                            python_model=CreditcardMlflowWrapper(),
                            artifacts={
                                "model_file": model_file,
                                "training_data": training_data, 
                                "testing_data": testing_data, 
                                "metadata_file": metadata_file,
                            }
                            )
    
    with open(metadata_file) as f:
        metadata = json.load(f)
    mlflow.log_params(metadata)