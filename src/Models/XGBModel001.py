import mlflow
import xgboost as xgb

from Models.BaseModel import BaseModel

'''
This is a basic, dumb model for proof-of-concept with comparing different models.
'''


class XGBModel001(BaseModel):
    def __init__(self, params):
        super().__init__("XGBModel001")
        self.params = params

    def train(self, X_train, y_train):
        with mlflow.start_run(run_name=self.model_name):
            self.model = xgb.XGBClassifier(**self.params)
            self.model.fit(X_train, y_train)
            mlflow.log_params(self.params)
            self.log_mlflow()

    def predict(self, x_test):
        return self.model.predict(x_test)

    def log_mlflow(self):
        mlflow.xgboost.log_model(self.model, self.model_name)
