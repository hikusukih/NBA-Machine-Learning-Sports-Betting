
import mlflow
from sklearn.neural_network import MLPClassifier

from Models.BaseModel import BaseModel


class NNModel001(BaseModel):
    def __init__(self, params):
        super().__init__("NNModel001")
        self.params = params

    def train(self, X_train, y_train):
        with mlflow.start_run(run_name=self.model_name):
            self.model = MLPClassifier(**self.params)
            self.model.fit(X_train, y_train)
            mlflow.log_params(self.params)
            self.log_model()

    def predict(self, X_test):
        return self.model.predict(X_test)

    def log_model(self):
        mlflow.sklearn.log_model(self.model, "neuralnet_001_model")
