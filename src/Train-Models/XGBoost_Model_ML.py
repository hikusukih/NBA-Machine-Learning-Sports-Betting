import sqlite3

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import warnings
# Filter out FutureWarnings from xgboost
warnings.filterwarnings("ignore", category=FutureWarning, module='xgboost.*')

dataset = "dataset_2012-24"
con = sqlite3.connect("../../Data/dataset.sqlite")
data = pd.read_sql_query(f"select * from \"{dataset}\"", con, index_col="index")
con.close()

margin = data['Home-Team-Win']
data.drop(['Score', 'Home-Team-Win', 'TEAM_NAME', 'Date', 'TEAM_NAME.1', 'Date.1', 'OU-Cover', 'OU'],
          axis=1, inplace=True)

data = data.values

data = data.astype(float)
acc_results = []

with tqdm(total=300) as progress_bar:
    for x in range(300):
        x_train, x_test, y_train, y_test = train_test_split(data, margin, test_size=.1)

        train = xgb.DMatrix(x_train, label=y_train)
        test = xgb.DMatrix(x_test, label=y_test)

        param = {
            "max_depth": 3,
            "learning_rate": 0.1,
            # "n_estimators": 200,
            "subsample": 0.6,
            "colsample_bytree": 0.6,
            "min_child_weight": 4,

            'eta': 0.01,
            # 'n_estimators': 300,
            'num_class': 2,
            'objective': 'multi:softprob',

        }
        epochs = 800

        model = xgb.train(param, train, epochs)
        predictions = model.predict(test)
        y = []

        for z in predictions:
            y.append(np.argmax(z))

        acc = round(accuracy_score(y_test, y) * 100, 1)
        # print(f"{acc}%")
        acc_results.append(acc)
        # only save results if they are the best so far
        best_acc = max(acc_results)
        if acc == best_acc:
            model.save_model('../../Models/XGBoost_{}%_ML-4.json'.format(acc))

        progress_bar.set_description(f"Accuracy: {best_acc}")
        progress_bar.update(1)
