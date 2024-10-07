import mlflow
from mlflow.tracking import MlflowClient

from Models.BaseModel import BaseModel
from data_cuts.base_data_cut import BaseDataCut


class ModelDataTrainingRun:
    def __init__(self, p_model: BaseModel, p_data_cut: BaseDataCut):
        if not isinstance(p_model, BaseModel):
            raise TypeError(TypeError("Parameter [p_model] must be of type BaseModel"))
        if not isinstance(p_data_cut, BaseDataCut):
            raise TypeError(TypeError("Parameter [p_data_cut] must be of type BaseDataCut"))

        self.my_model = p_model
        self.my_data_cut = p_data_cut

    def get_name(self):
        return self.my_model.get_name() + "_" + self.my_data_cut.get_name()

    def train_and_evaluate(self):

        with ((mlflow.start_run())):
            mlflow.log_params(self.my_params)
            mlflow.log_param("data_cut_type", self.my_data_cut.get_name())

            x_train = self.my_data_cut.get_x_train_data()
            x_test = self.my_data_cut.get_x_test_data()
            y_train = self.my_data_cut.get_y_train_data()
            y_test = self.my_data_cut.get_y_test_data()

            self.my_model.train(x_train, y_train)
            accuracy = self.my_model.evaluate(x_test, y_test)

            mlflow.log_metric("accuracy", accuracy)
            mlflow.sklearn.log_model(self.my_model, "model")
            mlflow.end_run()
        return accuracy
