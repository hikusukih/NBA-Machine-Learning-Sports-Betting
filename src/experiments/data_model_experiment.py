import mlflow
from mlflow.tracking import MlflowClient


class DataModelExperiment:
    def __init__(self, experiment_name):
        mlflow.set_experiment(experiment_name)
        self.client = MlflowClient()

    def run_experiment(self, model, data_cut, params):
        with mlflow.start_run():
            mlflow.log_params(params)
            mlflow.log_param("data_cut", data_cut.get_name())

            X, y = data_cut.get_data()
            X_train, X_test, y_train, y_test = self.preprocess_data(X, y)

            model.train(X_train, y_train)
            accuracy = model.evaluate(X_test, y_test)

            mlflow.log_metric("accuracy", accuracy)
            mlflow.sklearn.log_model(model, "model")

        return accuracy

    def preprocess_data(self, X, y):
        # Implement data preprocessing
        # This could include train-test split, scaling, etc.
        pass
