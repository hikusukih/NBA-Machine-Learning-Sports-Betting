import mlflow
from mlflow.models import infer_signature
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, fl_score

mlflow.set_tracking_uri="http://localhost:2048"

x,y = datasets.load_iris(return_X_y=True)