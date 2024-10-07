from abc import ABC, abstractmethod
import mlflow
import pandas as pd

class BaseModel(ABC):
    def __init__(self, model_name: str):
        self.model_name: str = model_name
        self.model = None


    def get_name(self) -> str:
        return self.model_name

    @abstractmethod
    def train(self, x_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Train the model on the given training data
        :param x_train: pd.DataFrame - Training data features, shape (n_samples, n_features)
        :param y_train: pd.Series - Training data labels, shape (n_samples,)
        :return: None
        """
        pass

    @abstractmethod
    def predict(self, x_test: pd.Series) -> pd.Series:
        """
        Apply the model to this test data
        :param x_test: pd.Series - Input test data, shape (n_samples,)
        :return: pd.Series - Results of applying the model, shape (n_samples,)
        """
        pass

    @abstractmethod
    def log_model(self) -> None:
        """
        Log all of your metrics, and a copy of the model for posterity.
        :return: None
        """
        pass
