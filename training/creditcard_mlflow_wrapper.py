import mlflow
from creditcard.models.creditcard_model import CreditcardModel

class CreditcardMlflowWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        model_file = context.artifacts["model_file"]
        self.creditcard_model = CreditcardModel(model_file)

    def predict(self, context, model_input_dict):
        prediction = self.creditcard_model.predict(**model_input_dict)
        return prediction