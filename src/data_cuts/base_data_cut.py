from abc import ABC, abstractmethod
import pandas as pd
from typing import Any


class BaseDataCut(ABC):
    def __init__(self,
                 feature_data: pd.DataFrame,
                 label_data: pd.Series,
                 test_size: float,
                 random_state: Any):
        """
        Initialize DataCut.

        :param feature_data: pd.DataFrame - Features to train on, shape (n_samples, n_features)
        :param label_data: pd.Series - Labels coinciding with the features, shape (n_samples,)
        :param test_size: float - Fraction of data to use as test data
        :param random_state: Any - Random state for reproducibility
        """

    def get_name(self) -> str:
        """Get a descriptive name for this DataCut

        :return str - The name of this instance
        """
        pass

    @abstractmethod
    def get_x_train_data(self) -> pd.DataFrame:
        """
        Get the training features data
        :return: pd.DataFrame - Training features, shape (n_samples_train, n_features)
        """
        pass

    @abstractmethod
    def get_x_test_data(self) -> pd.DataFrame:
        """
        Get the testing features data
        :return: pd.DataFrame - Testing features, shape (n_samples_test, n_features)
        """
        pass

    @abstractmethod
    def get_y_train_data(self):
        """
        Get the training labels data
        :return: pd.Series - Training labels, shape (n_samples_train,)
        """
        pass

    @abstractmethod
    def get_y_test_data(self):
        """
        Get the testing labels data
        :return: pd.Series - Training labels, shape (n_samples_train,)
        """
        pass

    @abstractmethod
    def log_mlflow(self):
        pass
