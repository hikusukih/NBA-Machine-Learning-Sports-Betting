# data_cuts/full_season_cut.py
from data_cuts.base_data_cut import BaseDataCut

class AllAvailableData(BaseDataCut):
    def __init__(self):
        self.season = season

    def get_name(self):
        return f"All Available Data"

    def get_x_train_data(self):
        pass

    def get_x_test_data(self):
        pass

    def get_y_train_data(self):
        pass

    def get_y_test_data(self):
        pass