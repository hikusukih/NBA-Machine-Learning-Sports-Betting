import pandas as pd
import numpy as np
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

class DataProcessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
        self.X = None
        self.y = None

    def load_data(self):
        # Load the sqlite db file
        dataset = "dataset_2012-24"
        con = sqlite3.connect(self.data_path)
        self.data = pd.read_sql_query(f"select * from \"{dataset}\" order by date", con, index_col="index")
        con.close()
        print(f"Data loaded. Shape: {self.data.shape}")

    def preprocess_data(self):
        # Assuming 'home_team_win' is the target variable
        self.y = self.data['Home-Team-Win']

        # features = ['home_team_rating', 'away_team_rating', 'home_team_form', 'away_team_form',
        #             'home_team_rest_days', 'away_team_rest_days', 'home_team_injuries', 'away_team_injuries']
        features = [
            'GP', 'W', 'L', 'W_PCT', 'MIN', 'FGM', 'FGA', 'FG_PCT',
            'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT',
            'OREB', 'DREB', 'REB', 'AST', 'TOV', 'STL', 'BLK',
            'PF', 'PFD', 'PTS', 'PLUS_MINUS', 'Days-Rest-Home', 'Days-Rest-Away'
        ]

        # Select features for prediction
        self.X = self.data[features]

        # Handle missing values
        imputer = SimpleImputer(strategy='mean')
        self.X = pd.DataFrame(imputer.fit_transform(self.X), columns=self.X.columns)

        # Normalize numerical features
        scaler = StandardScaler()
        self.X = pd.DataFrame(scaler.fit_transform(self.X), columns=self.X.columns)

        print("Data preprocessed.")

    def split_data(self, test_size=0.2, random_state=42):
        return train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)
