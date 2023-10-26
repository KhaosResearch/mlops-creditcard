import json
import os

import mlflow
import pandas as pd
from confluent_kafka import Producer
from confluent_kafka.error import ProduceError

from creditcard.models.creditcard_model import CreditcardModel

class CreditcardSeldonWrapper:

    def __init__(self, model_name, model_version="latest"):
        self.model_name = model_name
        self.model_version = model_version
        self.ready = False

        self.kafka_conf =  {
            'bootstrap.servers': '192.168.219.71:32003',
            'security.protocol': 'PLAINTEXT',
        }
    
    def connect_to_kafka(self):
        self.producer = Producer(self.kafka_conf)

    def load_model(self):
        model_uri = f"models:/{self.model_name}/{self.model_version}"
        download_path = mlflow.artifacts.download_artifacts(artifact_uri=model_uri)

        model_file = os.path.join(download_path, "artifacts", "model.joblib")

        self.predictor = CreditcardModel(model_file=model_file)

        self.ready = True

    def predict(self, X, features_names=None):
        if not self.ready:
            self.load_model()
            self.connect_to_kafka()
        
        if features_names is not None:
            data = pd.DataFrame(X, columns=features_names)
        else:
            data = X
        prediction = self.predictor.predict(data)
        predicted_label = self.predictor.prediction_to_label(int(prediction))

        try:
            value = json.dumps({"X":X,"Y":prediction})
            self.producer.produce(self.model_name, value=value)
        except ProduceError as e:
            print(f"Error producing {value} to kafka, it was not sent. Error trace:\n", e)
            pass

        return {'result' : predicted_label}
