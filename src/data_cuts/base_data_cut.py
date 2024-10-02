from abc import ABC, abstractmethod

class BaseDataCut(ABC):

    @abstractmethod
    def get_name(self):
        pass

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