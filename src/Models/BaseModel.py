from abc import ABC, abstractmethod
import mlflow

class BaseModel(ABC):
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None

    @abstractmethod
    def train(self, x_train, y_train):
        pass

    @abstractmethod
    def predict(self, x_test):
        pass

    @abstractmethod
    def log_model(self):
        pass
        # mlflow.pyfunc.log_model(self.model, self.model_name)
        mlflow.sklearn.log_model(self.model, "model")
