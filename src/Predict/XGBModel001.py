
import mlflow
import xgboost as xgb

from Predict.BaseModel import BaseModel


class XGBModel001(BaseModel):
    def __init__(self, params):
        super().__init__("XGBModel001")
        self.params = params

    def train(self, X_train, y_train):
        with mlflow.start_run(run_name=self.model_name):
            self.model = xgb.XGBClassifier(**self.params)
            self.model.fit(X_train, y_train)
            mlflow.log_params(self.params)
            self.log_model()

    def predict(self, X_test):
        return self.model.predict(X_test)
