
import mlflow
from sklearn.neural_network import MLPClassifier

from Models.BaseModel import BaseModel


class NNModel001(BaseModel):
    def __init__(self, params):
        super().__init__("NNModel001")
        self.params = params

    def train(self, X_train, y_train):
        self.model = MLPClassifier(**self.params)
        self._history = self.model.fit(X_train, y_train)
        mlflow.log_params(self.params)
        self.log_mlflow()

    def predict(self, X_test):
        return self.model.predict(X_test)

    def log_mlflow(self):
        mlflow.sklearn.log_model(self.model, self.model_name)
    # Log metrics for each epoch (training and validation loss/accuracy)
    #     for epoch in range(len(self._history.history['loss'])):
    #         mlflow.log_metric("train_loss", self._history.history['loss'][epoch], step=epoch)
    #         mlflow.log_metric("val_loss", self._history.history['val_loss'][epoch], step=epoch)
    #         mlflow.log_metric("train_accuracy", self._history.history['accuracy'][epoch], step=epoch)
    #         mlflow.log_metric("val_accuracy", self._history.history['val_accuracy'][epoch], step=epoch)


