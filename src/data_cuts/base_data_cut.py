from abc import ABC, abstractmethod

import mlflow
import pandas as pd


class BaseDataCut(ABC):
    """
    Initialize DataCut.

    :param feature_data: pd.DataFrame - Features to train on, shape (n_samples, n_features)
    :param label_data: pd.Series - Labels coinciding with the features, shape (n_samples,)
    """
    def __init__(self,
                 feature_data: pd.DataFrame,
                 label_data: pd.Series,
                 chain_parent: str = ""):
        self.feature_data = feature_data
        self.label_data = label_data
        self.chain_parent = chain_parent

    def get_name(self) -> str:
        """Get a descriptive name for this DataCut

        :return str - The name of this instance
        """
        pass

    @abstractmethod
    def get_processed_feature_data(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_processed_label_data(self) -> pd.Series:
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
    def get_y_train_data(self) -> pd.Series:
        """
        Get the training labels data
        :return: pd.Series - Training labels, shape (n_samples_train,)
        """
        pass

    @abstractmethod
    def get_y_test_data(self) -> pd.Series:
        """
        Get the testing labels data
        :return: pd.Series - Training labels, shape (n_samples_train,)
        """
        pass

    def log_mlflow(self) -> None:
        current_chain = ""
        if self.chain_parent != "":
            current_chain = self.chain_parent
        new_chain = f"{current_chain} > {self.get_name()}" if current_chain else self.get_name()
        mlflow.log_param("data_cut_chain", new_chain)
