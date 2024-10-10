import pandas as pd

from data_cuts.base_data_cut import BaseDataCut
from sklearn.model_selection import train_test_split


class AllAvailableData(BaseDataCut):
    def __init__(self,
                 feature_data: pd.DataFrame,
                 label_data: pd.Series,
                 chain_parent: str = "",
                 test_size=0.2, random_state=42):
        super().__init__(feature_data=feature_data,
                         label_data=label_data,
                         chain_parent=chain_parent)
        self.feature_data = feature_data
        self.label_data = label_data
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            feature_data.drop(["Date"], axis=1), label_data, test_size=test_size, random_state=random_state)

    def get_name(self) -> str:
        return "All_Available_Data"

    def get_processed_feature_data(self) -> pd.DataFrame:
        return self.feature_data

    def get_processed_label_data(self) -> pd.Series:
        return self.label_data

    def get_x_train_data(self):
        return self.x_train

    def get_x_test_data(self):
        return self.x_test

    def get_y_train_data(self):
        return self.y_train

    def get_y_test_data(self):
        return self.y_test
