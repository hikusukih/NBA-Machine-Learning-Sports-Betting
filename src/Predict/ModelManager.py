import mlflow


class ModelManager:
    def __init__(self):
        self.models = {}

    def add_model(self, model):
        self.models[model.model_name] = model

    def train_all_models(self, X_train, y_train):
        for model in self.models.values():
            model.train(X_train, y_train)

    def evaluate_models(self, X_test, y_test):
        results = {}
        for name, model in self.models.items():
            predictions = model.predict(X_test)
            accuracy = (predictions == y_test).mean()
            results[name] = accuracy
            mlflow.log_metric(f"{name}_accuracy", accuracy)
        return results