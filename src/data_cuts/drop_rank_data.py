import pandas as pd
from data_cuts.base_data_cut import BaseDataCut
from sklearn.model_selection import train_test_split


class DropRankFeatures(BaseDataCut):
    def __init__(self,
                 feature_data: pd.DataFrame,
                 label_data: pd.Series,
                 chain_parent: str = "",
                 test_size=0.2, random_state=42):
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
        self.drop_rank_data()
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.feature_data.drop(["Date"], axis=1),
            self.label_data,
            test_size=test_size, random_state=random_state)

    def get_name(self):
        return f"DropRankData"

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

    def drop_rank_data(self):
        labels_to_drop = [
            "GP_RANK",
            "W_RANK",
            "L_RANK",
            "W_PCT_RANK",
            "MIN_RANK",
            "FGM_RANK",
            "FGA_RANK",
            "FG_PCT_RANK",
            "FG3M_RANK",
            "FG3A_RANK",
            "FG3_PCT_RANK",
            "FTM_RANK",
            "FTA_RANK",
            "FT_PCT_RANK",
            "OREB_RANK",
            "DREB_RANK",
            "REB_RANK",
            "AST_RANK",
            "TOV_RANK",
            "STL_RANK",
            "BLK_RANK",
            "BLKA_RANK",
            "PF_RANK",
            "PFD_RANK",
            "PTS_RANK",
            "PLUS_MINUS_RANK"
        ]

        self.feature_data.drop(labels_to_drop, axis=1, inplace=True, errors='ignore')
