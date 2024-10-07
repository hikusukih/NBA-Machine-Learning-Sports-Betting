
import mlflow
import xgboost as xgb
from Models.BaseModel import BaseModel


import copy
from datetime import datetime

import numpy as np
import pandas as pd
import mlflow.xgboost


import sqlite3

from sklearn.metrics import accuracy_score,roc_curve, auc

from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
from tqdm import tqdm

import warnings


'''
This is the XGBoost_Runner model maker that came with the github project.
It's (in process of becoming) in the format of the 2024 way of comparing models 
'''


class XGBModel002(BaseModel):

    def log_mlflow(self):
        mlflow.xgboost.logModel(self.model, "xgboost_inherited_002_model")
        pass

    def __init__(self, params):
        super().__init__("XGBModel002")
        self.params ={
            "max_depth": 3,
            "learning_rate": 0.1,
            # "n_estimators": 200,
            "subsample": 0.8,
            "colsample_bytree": 0.6,
            "min_child_weight": 4,
            'eta': 0.01,
            # 'n_estimators': 300,
            'num_class': 2,
            'objective': 'multi:softprob',

            'tree_method': 'gpu_hist',
            'epochs': 800
        }

    def train(self, X_train, y_train):

        with mlflow.start_run(run_name=self.model_name):
            ## in original, this all was looped "iterations" (3) times

            train = xgb.DMatrix(X_train, label=y_train)
            # test = xgb.DMatrix(x_test, label=y_test)

            self.model = xgb.train(self.params, train, self.params['epochs'])

            self.model = xgb.XGBClassifier(**self.params)
            self.model.fit(X_train, y_train)
            mlflow.log_params(self.params)
            self.log_mlflow()

    # def predict(self, x_test):
    #     return self.model.predict(x_test)


# Filter out FutureWarnings from xgboost
# warnings.filterwarnings("ignore", category=FutureWarning, module='xgboost.*')

# Parameters!
split_date = pd.Timestamp('2023-08-01')
iterations = 3
param = {
    "max_depth": 3,
    "learning_rate": 0.1,
    # "n_estimators": 200,
    "subsample": 0.8,
    "colsample_bytree": 0.6,
    "min_child_weight": 4,
    'eta': 0.01,
    # 'n_estimators': 300,
    'num_class': 2,
    'objective': 'multi:softprob',

    'tree_method': 'gpu_hist'
}

# Load Data
dataset = "dataset_2012-24"
con = sqlite3.connect("../../Data/dataset.sqlite")
data = pd.read_sql_query(f"select * from \"{dataset}\" order by date", con, index_col="index")
con.close()

# Won't need these for anything
data.drop(['Score',
           # 'Home-Team-Win',
           'TEAM_NAME',
           'TEAM_NAME.1',
           'Date.1',
           'OU-Cover', 'OU'],
          axis=1, inplace=True)

# margin = data['Home-Team-Win']

# print(data.columns)
# data = data.astype(float)

# Identify columns that are not 'Date' and are numeric
numeric_cols = data.select_dtypes(include=['number']).columns.tolist()

# Convert these columns to float
data[numeric_cols] = data[numeric_cols].astype(float)
data['Date'] = pd.to_datetime(data['Date'])
# data = data.values

acc_results = []
# prob_true_list = []
# prob_pred_list = []

model_labels = []
all_prob_true = []
all_prob_pred = []


train_data = data[data['Date'] < split_date]
test_data = data[data['Date'] >= split_date]

x_train = train_data.drop(['Home-Team-Win', 'Date'], axis=1)
y_train = train_data['Home-Team-Win']
x_test = test_data.drop(['Home-Team-Win', 'Date'], axis=1)
y_test = test_data['Home-Team-Win']

with tqdm(total=iterations) as progress_bar:
    for x in range(iterations):
        # x_train, x_test, y_train, y_test = train_test_split(data, margin, test_size=.1)


        train = xgb.DMatrix(x_train, label=y_train)
        test = xgb.DMatrix(x_test, label=y_test)

        epochs = 800

        model = xgb.train(param, train, epochs)
        predictions = model.predict(test)
        y = []

        # Get the predicted probabilities for the positive class
        y_probs = predictions[:, 1]  # Assuming 1 is the positive class


        for z in predictions:
            y.append(np.argmax(z))
            y_pred = np.argmax(predictions, axis=1)

        acc = round(accuracy_score(y_test, y) * 100, 1)
        # print(f"{acc}%")
        acc_results.append(acc)
        # only save results if they are the best so far
        best_acc = max(acc_results)

        prob_true, prob_pred = calibration_curve(y_test, y_probs, n_bins=10)
        model_labels.append(f"M_{acc}%")
        all_prob_true.append(prob_true)
        all_prob_pred.append(prob_pred)
        if acc == best_acc:
            model.save_model('../../Models/KronosModel/XGBoost_{}%_ML-4.json'.format(acc))

        progress_bar.set_description(f"Accuracy: {best_acc}")
        progress_bar.update(1)
print('Done Training!')
# Plotting all calibration curves
plt.figure(figsize=(10, 8))
for i in range(len(all_prob_true)):
    plt.plot(all_prob_pred[i], all_prob_true[i], marker='o', linestyle='-', linewidth=1, label=model_labels[i])
print('Done All Prob Listing!')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Ideally Calibrated')
print('Done Plotting!')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration Curves for All Models')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Adjust legend location to avoid blocking the graph
plt.grid(True)
plt.show()
print("If you don't see it, look for the plot on another monitor.")

# # # Plotting the calibration curve - Average all "BEST"s
# if len(prob_true_list) > 0:
#     mean_prob_true = np.mean(prob_true_list, axis=0)
#     mean_prob_pred = np.mean(prob_pred_list, axis=0)
#     plt.figure(figsize=(8, 6))
#     plt.plot(mean_prob_pred, mean_prob_true, marker='o', linewidth=1, label='Calibration plot')
#     plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
#     plt.xlabel('Average predicted probability in each bin')
#     plt.ylabel('Fraction of positives')
#     plt.title('Calibration Curve')
#     plt.legend()
#     plt.show()

# # Plotting the calibration curve - for the BEST ONLY
# plt.figure(figsize=(8, 6))
# plt.plot(prob_pred_list, prob_true_list, marker='o', linewidth=1, label='Calibration plot')
# plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
# plt.xlabel('Average predicted probability in each bin')
# plt.ylabel('Fraction of positives')
# plt.title('Calibration Curve')
# plt.legend()
# plt.show()



# Assume you have y_test and y_scores from your model
fpr, tpr, thresholds = roc_curve(y_test, predictions[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()