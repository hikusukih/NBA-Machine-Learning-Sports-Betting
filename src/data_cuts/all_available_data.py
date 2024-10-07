# data_cuts/full_season_cut.py
import mlflow

from data_cuts.base_data_cut import BaseDataCut
from sklearn.model_selection import train_test_split


class AllAvailableData(BaseDataCut):
    def __init__(self, feature_data, label_data, test_size, random_state, data_cut_name="all_available_data"):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            feature_data, label_data, test_size=test_size, random_state=random_state)

    def get_x_train_data(self):
        return self.x_train

    def get_x_test_data(self):
        return self.x_test

    def get_y_train_data(self):
        return self.y_train

    def get_y_test_data(self):
        return self.y_test

    def log_mlflow(self):
        mlflow.log_param("data_cut_name", self.get_name())
