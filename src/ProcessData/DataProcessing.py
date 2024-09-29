import pandas as pd
import numpy as np
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
        # Load the CSV file
        self.data = pd.read_csv(self.data_path)
        print(f"Data loaded. Shape: {self.data.shape}")

    def preprocess_data(self):
        # Assuming 'home_team_win' is the target variable
        self.y = self.data['home_team_win']

        # Select features for prediction
        features = ['home_team_rating', 'away_team_rating', 'home_team_form', 'away_team_form',
                    'home_team_rest_days', 'away_team_rest_days', 'home_team_injuries', 'away_team_injuries']
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
