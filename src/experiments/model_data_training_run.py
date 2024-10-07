import mlflow

from Models.BaseModel import BaseModel
from data_cuts.base_data_cut import BaseDataCut


class ModelDataTrainingRun:
    def __init__(self, p_model: BaseModel, p_data_cut: BaseDataCut):
        if not isinstance(p_model, BaseModel):
            raise TypeError(TypeError("Parameter [p_model] must be of type BaseModel"))
        if not isinstance(p_data_cut, BaseDataCut):
            raise TypeError(TypeError("Parameter [p_data_cut] must be of type BaseDataCut"))

        self._model = p_model
        self._data_cut = p_data_cut

    def get_name(self):
        return self._model.get_name() + "_" + self._data_cut.get_name()

    def train_and_evaluate(self):
        with ((mlflow.start_run())):
            mlflow.log_param("data_cut_name", self._data_cut.get_name())

            x_train = self._data_cut.get_x_train_data()
            x_test = self._data_cut.get_x_test_data()
            y_train = self._data_cut.get_y_train_data()
            y_test = self._data_cut.get_y_test_data()

            self._model.train(x_train, y_train)
            predictions = self._model.predict(x_test)
            accuracy = (predictions == y_test).mean()
            accuracy = self._model.evaluate(x_test, y_test)

            mlflow.log_metric("accuracy", accuracy)
            mlflow.sklearn.log_model(self._model, "model")
            mlflow.end_run()
        return accuracy
