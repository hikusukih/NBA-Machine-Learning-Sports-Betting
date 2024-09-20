import itertools
import random
import sqlite3

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.base import BaseEstimator, clone
from tqdm import tqdm
import json

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module='xgboost.*')

n_iter = 100

# Load your data
dataset = "dataset_2012-24"
con = sqlite3.connect("../../Data/dataset.sqlite")
data = pd.read_sql_query(f"SELECT * FROM \"{dataset}\" order by Date", con, index_col="index")
con.close()

# Prepare your data
margin = data['Home-Team-Win']
data.drop(['Score',
           # 'Home-Team-Win',
           'TEAM_NAME',
           # 'Date',
           'TEAM_NAME.1', 'Date.1',
           'OU-Cover', 'OU'], axis=1, inplace=True)
# data = data.values.astype(float)

data['Date'] = pd.to_datetime(data['Date'])
# Split the dataset into training and testing sets
# x_train, x_test, y_train, y_test = train_test_split(data, margin, test_size=0.1, random_state=32)


# Initialize XGBoost Classifier
# xgb_clf = xgb.XGBClassifier(objective='binary:logistic')

# Set up RandomizedSearchCV
# random_search = RandomizedSearchCV(xgb_clf, param_distributions=param_grid, n_iter=n_iter, scoring='accuracy', cv=5, verbose=3, random_state=42)

# Initialize variables to track the best model
# best_score = -np.inf
# best_params = None

# paramodel_iterator = 0

# print(f'Here goes!!! Best: {best_score}')
# # Run RandomizedSearchCV
# for i in range(random_search.n_iter):
#     random_search.fit(x_train, y_train)
#     current_score = random_search.best_score_
#     current_params = random_search.best_params_
#
#     if current_score > best_score:
#         best_score = current_score
#         best_params = current_params
#         best_model = random_search.best_estimator_
#
#         best_model_so_far = f'../../Models/best_xgb_model_{paramodel_iterator}.json'
#         best_parameters_so_far = f'../../Models/best_xgb_parameters_{paramodel_iterator}.json'
#         paramodel_iterator += 1
#
#         best_model.save_model(best_model_so_far)
#         with open(best_parameters_so_far, 'w') as f:
#             json.dump(best_params, f, indent=4)
#             print(f"New best model with score: {best_score}")

# Define the hyperparameter grid
param_grid = {
    'max_depth': [3,6,4, 5 ,7],
    'learning_rate': [ 0.1, 0.05,0.01],
    'n_estimators': [200, 300,100],
    'subsample': [0.8, 0.7, 0.9, 0.6],
    'colsample_bytree': [0.55 , 0.8, 0.7, 0.95],
    'min_child_weight': [4, 3, 1, 2],
    'eta': [.01, .015],
    'split_date': [pd.Timestamp('2023-12-01'),
                   pd.Timestamp('2023-08-01')],
    'tree_method': ['gpu_hist', 'approx', 'hist']
}

# Create all combinations of parameters
all_params_combos = [dict(zip(param_grid, v)) for v in itertools.product(*param_grid.values())]
random.shuffle(all_params_combos)
# Initialize variables to track the best model
best_score = -np.inf
best_params = None

# possibleCombos = (len(param_grid['max_depth']) * len(param_grid['learning_rate'])
#                   * len(param_grid['n_estimators']) * len(param_grid['subsample'])
#                   * len(param_grid['colsample_bytree']) * len(param_grid['split_date'])
#                   * len(param_grid['min_child_weight']))
with tqdm(total=len(all_params_combos)) as progress_bar:
    # Iterate over the grid
    for params in all_params_combos:

    # for max_depth in param_grid['max_depth']:
    #     for learning_rate in param_grid['learning_rate']:
    #         for n_estimators in param_grid['n_estimators']:
    #             for subsample in param_grid['subsample']:
    #                 for colsample_bytree in param_grid['colsample_bytree']:
    #                     for split_date in param_grid['split_date']:
        train_data = data[data['Date'] < params['split_date']]
        test_data = data[data['Date'] >= params['split_date']]

        x_train = train_data.drop(['Home-Team-Win', 'Date'], axis=1)
        y_train = train_data['Home-Team-Win']
        x_test = test_data.drop(['Home-Team-Win', 'Date'], axis=1)
        y_test = test_data['Home-Team-Win']
                            # for min_child_weight in param_grid['min_child_weight']:
                            #     progress_bar.set_description(f'md{max_depth}lr{learning_rate}es{n_estimators}ss{subsample}cb{colsample_bytree}cw{min_child_weight}')

        model = xgb.XGBClassifier(max_depth=params['max_depth'],
                                  learning_rate=params['learning_rate'],
                                  n_estimators=params['n_estimators'],
                                  subsample=params['subsample'],
                                  colsample_bytree=params['colsample_bytree'],
                                  min_child_weight=params['min_child_weight'],
                                  objective='binary:logistic')

        # Create and fit the model
        model.fit(x_train, y_train)

        # Evaluate the model
        y_pred = model.predict(x_test)
        current_score = accuracy_score(y_test, y_pred)

        # Update the best model if current model is better
        if current_score > best_score:
            best_score = current_score

            # Adjust the 'split_date' in params for JSON serialization
            params_for_json = params.copy()
            params_for_json['split_date'] = params['split_date'].strftime('%Y-%m-%d')

            best_params = params
            # {'max_depth': max_depth, 'learning_rate': learning_rate,
            #                'n_estimators': n_estimators, 'subsample': subsample,
            #                'colsample_bytree': colsample_bytree, 'min_child_weight': min_child_weight,
            #                'split_date': split_date.strftime('%Y-%m-%d %H:%M:%S')}
            model.save_model(f'../../Models/XGBoost_Model_{best_score*100}%_ML.json')
            best_model = model
            with open(f'../../Models/XGBoost_params_{best_score*100}%_ML.json', 'w') as f:
                json.dump(params_for_json, f, indent=4)

            progress_bar.set_description(f'Current Best Score: {round(best_score*100,1)}')
        progress_bar.update(1)

# Print the best parameters and best score
print("Final Best Parameters:", best_params)
print("Final Best Score:", best_score)
