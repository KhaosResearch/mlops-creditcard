import joblib

class CreditcardModel:

    def __init__(self, model_file):
        self.model_file = model_file
        self.clf = joblib.load(model_file)

    def predict(self, X):
        y_pred = self.clf.predict(X)
        return y_pred
    
    def prediction_to_label(self, y_pred):
        return "fraudulent" if y_pred == 1 else "not_fraudulent"