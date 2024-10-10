import pandas as pd
from typing import Any
from data_cuts.base_data_cut import BaseDataCut
from sklearn.model_selection import train_test_split
from datetime import datetime


class DropSillyData(BaseDataCut):
    def __init__(self, feature_data: pd.DataFrame, label_data: pd.Series,
                 test_size=0.2, random_state=42, chain_parent: str = ""):
        """Drop features unlikely to impact training results. Very subjective.

        This is intended to provide passthrough data for other data_cuts but can give a random split too.

        :param season_start_year: The first year of the season - which October does it start in?
        :param feature_data: All the features we can work with
        :param label_data: Labels for the datapoints
        :param date_column_name: A column identifier for the date of the game being played
        """
        super().__init__(feature_data=feature_data, label_data=label_data, chain_parent=chain_parent)
        self.preprocess_data()
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            feature_data.drop(["Date"], axis=1), label_data, test_size=test_size, random_state=random_state)

    def preprocess_data(self):
        labels_to_drop = [
            "MIN",
            "FGM",
            "FG3M",
            "FTM",
            "GP_RANK",
            "W_RANK",
            "L_RANK",
            "MIN_RANK",
            "FGM_RANK",
            "FG3M_RANK",
            "FTM_RANK",
            "MIN.1",
            "FGM.1",
            "FG3M.1",
            "FTM.1",
            "GP_RANK.1",
            "W_RANK.1",
            "L_RANK.1",
            "MIN_RANK.1",
            "FGM_RANK.1",
            "FG3M_RANK.1",
            "FTM_RANK.1"
        ]

        self.feature_data.drop(labels_to_drop, axis=1, inplace=True, errors='ignore')

    def get_name(self):
        return f"DropSillyData"

    def get_x_train_data(self):
        return self.x_train

    def get_x_test_data(self):
        return self.x_test

    def get_y_train_data(self):
        return self.y_train

    def get_y_test_data(self):
        return self.y_test

    def get_processed_feature_data(self):
        return self.feature_data

    def get_processed_label_data(self):
        return self.label_data
