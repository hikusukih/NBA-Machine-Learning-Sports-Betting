from abc import ABC, abstractmethod
import mlflow

class BaseModel(ABC):
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None

    @abstractmethod
    def train(self, X_train, y_train):
        pass

    @abstractmethod
    def predict(self, X_test):
        pass

    def log_model(self):
        mlflow.log_model(self.model, self.model_name)