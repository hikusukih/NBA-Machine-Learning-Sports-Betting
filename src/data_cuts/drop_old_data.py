import pandas as pd
from data_cuts.base_data_cut import BaseDataCut
from sklearn.model_selection import train_test_split


class DropOldData(BaseDataCut):
    def __init__(self,
                 feature_data: pd.DataFrame,
                 label_data: pd.Series,
                 cutoff_year: int,
                 chain_parent: str = "",
                 test_size=0.2, random_state=42
                 ):
        """Drop features unlikely to impact training results. Very subjective.

        This is intended to provide passthrough data for other data_cuts but can give a random split too.

        :param season_start_year: The first year of the season - which October does it start in?
        :param feature_data: All the features we can work with
        :param label_data: Labels for the datapoints
        :param date_column_name: A column identifier for the date of the game being played
        """
        super().__init__(feature_data=feature_data,
                         label_data=label_data,
                         chain_parent=chain_parent)
        self.drop_old_data(cutoff_year=cutoff_year)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            feature_data.drop(["Date"], axis=1), label_data, test_size=test_size, random_state=random_state)

    def get_name(self):
        return f"DropOldData"

    def get_processed_feature_data(self):
        return self.feature_data

    def get_processed_label_data(self):
        return self.label_data

    def get_x_train_data(self):
        return self.x_train

    def get_x_test_data(self):
        return self.x_test

    def get_y_train_data(self):
        return self.y_train

    def get_y_test_data(self):
        return self.y_test

    def drop_old_data(self, cutoff_year: int):
        # October first of the year
        cutoff_date = pd.to_datetime(f"{cutoff_year}-10-01")

        self.feature_data['Date'] = pd.to_datetime(self.feature_data['Date'])
        self.feature_data = self.feature_data[self.feature_data['Date'] >= cutoff_date]
        # Remove the label data at the indices of the removed date data - can't remember how!
