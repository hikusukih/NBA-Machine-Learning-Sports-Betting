from data_cuts.base_data_cut import BaseDataCut

class AllDataEver(BaseDataCut):
    def __init__(self, season):
        self.season = season

    def get_data(self):
        # Implementation to fetch full season data
        pass

    def get_name(self):
        return f"All_Data_Ever"