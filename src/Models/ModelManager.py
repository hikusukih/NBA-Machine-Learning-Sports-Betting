import mlflow
import pandas as pd


class ModelManager:
    def __init__(self):
        self.models = {}

    def add_model(self, model):
        self.models[model.model_name] = model

    def train_all_models(self, x_train, y_train):
        for model in self.models.values():
            model.train(x_train, y_train)

    def evaluate_models(self, x_test, y_test):
        """

        :param x_test: Features for Testing. Shape: TODO
        :param y_test: Labels for testing. Shape: TODO
        :return:
        """
        results = {}

        # If y_test is a DataFrame or a 2D array, reduce it to a single column
        if isinstance(y_test, pd.DataFrame) or len(y_test.shape) > 1:
            y_test = y_test.iloc[:, 0]  # or y_test[:, 0] if it's a numpy array

        for name, model in self.models.items():
            predictions = model.predict(x_test)
            # Ensure predictions have the same shape

            if len(predictions.shape) > 1:
                predictions = predictions[:, 0]  # Reduce predictions to a single column if necessary

            accuracy = (predictions == y_test).mean()
            results[name] = accuracy
            mlflow.log_metric(f"{name}_accuracy", accuracy)
        return results
