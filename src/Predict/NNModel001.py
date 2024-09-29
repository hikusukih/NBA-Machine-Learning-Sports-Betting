import BaseModel
import mlflow
from sklearn.neural_network import MLPClassifier

class NeuralNetworkModel(BaseModel):
    def __init__(self, params):
        super().__init__("neural_network_model")
        self.params = params

    def train(self, X_train, y_train):
        with mlflow.start_run(run_name=self.model_name):
            self.model = MLPClassifier(**self.params)
            self.model.fit(X_train, y_train)
            mlflow.log_params(self.params)
            self.log_model()

    def predict(self, X_test):
        return self.model.predict(X_test)