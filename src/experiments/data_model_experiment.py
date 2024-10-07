import mlflow
from mlflow.tracking import MlflowClient

from Models.BaseModel import BaseModel
from data_cuts.base_data_cut import BaseDataCut


class DataModelExperiment:
    def __init__(self, experiment_name):
        mlflow.set_experiment(experiment_name)
        self.client = MlflowClient()

    def run_experiment(self, model, data_cut, params):
        if not isinstance(input, BaseModel):
            raise TypeError(TypeError("Parameter [model] must be of type BaseModel"))
        if not isinstance(input, BaseDataCut):
            raise TypeError(TypeError("Parameter [data_cut] must be of type BaseDataCut"))

        with ((mlflow.start_run())):
            mlflow.log_params(params)
            mlflow.log_param("data_cut_type", data_cut.get_name())

            x_train = data_cut.get_x_train_data()
            x_test = data_cut.get_x_test_data()
            y_train = data_cut.get_y_train_data()
            y_test = data_cut.get_y_test_data()

            model.train(x_train, y_train)
            accuracy = model.evaluate(x_test, y_test)

            mlflow.log_metric("accuracy", accuracy)
            mlflow.sklearn.log_model(model, "model")

        return accuracy

    def preprocess_data(self, X, y):
        # Implement data preprocessing
        # This could include train-test split, scaling, etc.
        pass
