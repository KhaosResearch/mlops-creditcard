import os

import mlflow
from creditcard.models.creditcard_model import CreditcardModel

class CreditcardSeldonWrapper:

    def __init__(self, model_name, model_version="latest"):
        self.model_name = model_name
        self.model_version = model_version
        self.ready = False

    def load_model(self):
        model_uri = f"models:/{self.model_name}/{self.model_version}"
        download_path = mlflow.artifacts.download_artifacts(artifact_uri=model_uri)

        model_file = os.path.join(download_path, "artifacts", "model.joblib")

        self.predictor = CreditcardModel(model_file=model_file)

        self.ready = True

    def predict(self, X, features_names=None):
        if not self.ready:
            self.load_model()
        
        prediction = self.predictor.predict(X)
        predicted_label = self.predictor.prediction_to_label(int(prediction))
        return {'result' : predicted_label}
