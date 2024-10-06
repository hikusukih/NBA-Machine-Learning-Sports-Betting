from abc import ABC, abstractmethod


class BaseDataCut(ABC):

    def __init__(self, feature_data, label_data, test_size, random_state, data_cut_name):
        """
        Initialize DataCut. Select which data does what here.
        :param feature_data: Features to train on
        :param label_data: Labels coinciding with the features
        :param data_cut_name: The name of this instance
        """
        self.random_state = random_state
        self.test_size = test_size
        self.name = data_cut_name
        self.feature_data = feature_data
        self.label_data = label_data

    def get_name(self):
        return self.name

    @abstractmethod
    def get_x_train_data(self):
        pass

    @abstractmethod
    def get_x_test_data(self):
        pass

    @abstractmethod
    def get_y_train_data(self):
        pass

    @abstractmethod
    def get_y_test_data(self):
        pass