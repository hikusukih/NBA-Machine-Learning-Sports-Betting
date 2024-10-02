import pandas as pd
import sqlite3

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer


class InitialDataProcessor:
    """
    This delivers the data in its least processed form.
     Downstream "DataCuts" will filter this down.
    """
    def __init__(self, data_path):
        """

        :param data_path: Path to sqlite database
        """
        # Path to sqlite database
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
        target_variable_column_name = 'Home-Team-Win'
        self.y = self.data[target_variable_column_name]

        # Select features for prediction
        # TODO: ensure all of the most-relevant columns are included
        # TODO: Maybe that's in a data cut ? This should literally be all features?
        features = [
            'GP', 'W', 'L', 'W_PCT', 'MIN', 'FGM', 'FGA', 'FG_PCT',
            'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT',
            'OREB', 'DREB', 'REB', 'AST', 'TOV', 'STL', 'BLK',
            'PF', 'PFD', 'PTS', 'PLUS_MINUS', 'Days-Rest-Home', 'Days-Rest-Away'
        ]
        team_name_1_column_name = 'TEAM_NAME'
        team_name_1_encoded_column_name = 'TEAM_NAME_Encoded'
        team_name_2_column_name = 'TEAM_NAME.1'
        team_name_2_encoded_column_name = 'TEAM_NAME_Encoded.1'
        team_encoder = LabelEncoder()

        self.data[team_name_1_encoded_column_name] = team_encoder.fit_transform(self.data[team_name_1_column_name])
        self.data[team_name_2_encoded_column_name] = team_encoder.fit_transform(self.data[team_name_2_column_name])

        team_encoder = LabelEncoder()
        self.data[team_name_1_encoded_column_name] = team_encoder.fit_transform(self.data[team_name_1_column_name])
        self.data[team_name_2_encoded_column_name] = team_encoder.fit_transform(self.data[team_name_2_column_name])

        date_column_name = 'Date'
        date_analog_column_name = 'DaysSince1990'
        date_1_column_name = 'Date.1'
        date_1_analog_column_name = 'DaysSince1990.1'

        reference_date = pd.Timestamp('1990-01-01')

        # make it a timestamp
        self.data[date_column_name] = pd.to_datetime(self.data[date_column_name], errors='coerce')
        self.data[date_1_column_name] = pd.to_datetime(self.data[date_1_column_name], errors='coerce')
        # Replace your date column with days since reference date
        self.data[date_analog_column_name] = (self.data[date_column_name] - reference_date).dt.days
        self.data[date_1_analog_column_name] = (self.data[date_1_column_name] - reference_date).dt.days

        # self.X = self.data[features]
        self.X = self.data.drop([target_variable_column_name,
                                 date_column_name,
                                 date_1_column_name,
                                 team_name_1_column_name,
                                 team_name_2_column_name], axis=1)

        # Handle missing values
        imputer = SimpleImputer(strategy='mean')
        self.X = pd.DataFrame(imputer.fit_transform(self.X), columns=self.X.columns)

        # Normalize numerical features
        scaler = StandardScaler()
        self.X = pd.DataFrame(scaler.fit_transform(self.X), columns=self.X.columns)

        print("Data preprocessed.")

    def split_data(self, test_size=0.2, random_state=42):
        print("InitialDataProcessor#split_data")
        print(self.y.shape) # Should be a one-dimenaional array with shape (2959,)

        # Ensure self.y is a single-column array or series
        if isinstance(self.y, pd.DataFrame):
            self.y = self.y.iloc[:, 0]  # Select the first column if y is a DataFrame


        return train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)
