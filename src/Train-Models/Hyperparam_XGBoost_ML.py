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
data = pd.read_sql_query(f"SELECT * FROM \"{dataset}\"", con, index_col="index")
con.close()

# Prepare your data
margin = data['Home-Team-Win']
data.drop(['Score', 'Home-Team-Win', 'TEAM_NAME', 'Date', 'TEAM_NAME.1', 'Date.1', 'OU-Cover', 'OU'], axis=1, inplace=True)
data = data.values.astype(float)

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, margin, test_size=0.1, random_state=32)


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
# Initialize variables to track the best model
best_score = -np.inf
best_params = None

# Define the hyperparameter grid
param_grid = {
    'max_depth': [#3,6,4,5,7,
        8],
    'learning_rate': [ 0.1, 0.05, 0.01],
    'n_estimators': [200, 400, 300, 100],
    'subsample': [0.8, 0.7, 0.9, 0.6],
    'colsample_bytree': [0.6, 0.8, 0.7, 0.9],
    'min_child_weight': [4, 3, 1, 2],

    'num_class': 2, # Always 2 - 'win' or 'lose'
    'eta': [.01, .005, .015],
}
possibleCombos = 6*3*4*4*4*4
with tqdm(total=possibleCombos) as progress_bar:
    # Iterate over the grid
    for max_depth in param_grid['max_depth']:
        for learning_rate in param_grid['learning_rate']:
            for n_estimators in param_grid['n_estimators']:
                for subsample in param_grid['subsample']:
                    for colsample_bytree in param_grid['colsample_bytree']:
                        for min_child_weight in param_grid['min_child_weight']:
                            progress_bar.set_description(f'md{max_depth}lr{learning_rate}es{n_estimators}ss{subsample}cb{colsample_bytree}cw{min_child_weight}')
                            progress_bar.update(1)
                            # Create and fit the model
                            model = xgb.XGBClassifier(max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimators, subsample=subsample, objective='binary:logistic')
                            model.fit(x_train, y_train)

                            # Evaluate the model
                            y_pred = model.predict(x_test)
                            current_score = accuracy_score(y_test, y_pred)

                            # Update the best model if current model is better
                            if current_score > best_score:
                                best_score = current_score
                                best_params = {'max_depth': max_depth, 'learning_rate': learning_rate,
                                               'n_estimators': n_estimators, 'subsample': subsample,
                                               'colsample_bytree': colsample_bytree, 'min_child_weight': min_child_weight}
                                model.save_model(f'../../Models/XGBoost_Model_{best_score*100}%_ML.json')

                                with open(f'../../Models/XGBoost_params_{best_score*100}%_ML.json', 'w') as f:
                                    json.dump(best_params, f, indent=4)



# [CV 5/5] END colsample_bytree=0.6, learning_rate=0.01, max_depth=3, min_child_weight=4, n_estimators=200, subsample=0.7;, score=0.670 total time=   0.6s

# [CV 5/5] END colsample_bytree=0.8, learning_rate=0.01, max_depth=3, min_child_weight=3, n_estimators=400, subsample=0.6;, score=0.668 total time=   1.2s

# [CV 5/5] END colsample_bytree=0.7, learning_rate=0.01, max_depth=3, min_child_weight=1, n_estimators=300, subsample=0.6;, score=0.667 total time=   0.9s
# [CV 5/5] END colsample_bytree=0.9, learning_rate=0.01, max_depth=3, min_child_weight=4, n_estimators=400, subsample=0.8;, score=0.667 total time=   1.3s

# [CV 5/5] END colsample_bytree=0.9, learning_rate=0.05, max_depth=3, min_child_weight=3, n_estimators=100, subsample=0.6;, score=0.665 total time=   0.3s
# [CV 5/5] END colsample_bytree=0.7, learning_rate=0.05, max_depth=3, min_child_weight=4, n_estimators=200, subsample=0.6;, score=0.665 total time=   0.6s
#
#
#
#





# Print the best parameters and best score
print("Final Best Parameters:", best_params)
print("Final Best Score:", best_score)

# Evaluate the best model on the test set
y_pred = best_model.predict(x_test)
print("Test Set Accuracy:", accuracy_score(y_test, y_pred))

# Save the best model
best_model.save_model(best_model_so_far)
