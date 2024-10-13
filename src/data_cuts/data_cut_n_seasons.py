import pandas as pd
from data_cuts.base_data_cut import BaseDataCut
from datetime import datetime


class DropSeasonsBeforeDate(BaseDataCut):
    """This is a "terminal" data_cut in that you consume its train/test/label data
    """

    def __init__(self,
                 feature_data: pd.DataFrame,
                 label_data: pd.Series,
                 season_start_year: int,
                 chain_parent: str = '',
                 date_column: str = 'Date'):
        super().__init__(feature_data=feature_data,
                         label_data=label_data,
                         chain_parent=chain_parent)
        self.season_start_year = season_start_year
        self.date_column = date_column
        self.x_train, self.x_test, self.y_train, self.y_test = self._season_split()

    def get_name(self):
        return f"ValidateOnSeasonsSince{self.season_start_year}"

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

    def _season_split(self):
        """Split the data into training and testing sets based on season or date.

        Parameters:
            - feature_data: The full feature data DataFrame
            - label_data: The corresponding labels for the feature data

        Returns:
            A tuple containing (x_train, x_test, y_train, y_test).
        """
        temp_date_column_name = "temp_date_column_name"
        # Convert the date column to datetime if not already
        df_mask = pd.DataFrame(self.feature_data)
        df_mask[temp_date_column_name] = pd.to_datetime(df_mask[self.date_column])

        # Create a datetime object for October 1 of the season year
        october_first = datetime(self.season_start_year, 10, 1)

        # Create a boolean mask for train/test based on the selected season/year
        train_mask = df_mask[temp_date_column_name] < october_first
        test_mask = df_mask[temp_date_column_name] >= october_first

        # Apply the mask to get train/test sets
        x_train, y_train = (self.feature_data[train_mask].drop([self.date_column], axis=1),
                            self.label_data[train_mask])
        x_test, y_test = (self.feature_data[test_mask].drop([self.date_column], axis=1),
                          self.label_data[test_mask])

        return x_train, x_test, y_train, y_test
