import pandas as pd
import sqlite3

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


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
        # Data as gathered from db
        self.raw_data = None
        # Processed data
        self.data = None
        # Pre-Processed Data
        self.X = None
        # Labels on Data
        self.y = None

    def load_data(self):
        # Load the sqlite db file
        dataset = "dataset_2012-24"
        con = sqlite3.connect(self.data_path)
        self.raw_data = pd.read_sql_query(f"select * from \"{dataset}\" order by date", con, index_col="index")
        con.close()
        self.data = self.raw_data
        print(f"Data loaded. Shape: {self.data.shape}")

    def preprocess_data(self):
        """Do broad processing to the test/train and label data
        1
        Any transformations will be stored as new columns

        Labels and predictors like Score will be removed from test/train data

        :return:
        """
        # 'home_team_win' is the moneyline target variable
        moneyline_target_column_name = 'Home-Team-Win'
        self.y = self.data[moneyline_target_column_name]

        score_column_name = 'Score'
        overunder_target_variable_column_name = 'OU-Cover'
        ou_determinant_column_name = 'OU'

        # Handle strings - they'll be turned into numbers.
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

        # Handle Dates - they'll be turned into a number of days
        date_column_name = 'Date'
        date_as_number_column_name = 'DaysSince1990'
        # This is a duplicate of 'Date' - just drop it.
        date_1_column_name = 'Date.1'

        reference_date = pd.Timestamp('1990-01-01')

        # make it a timestamp
        self.data[date_column_name] = pd.to_datetime(self.data[date_column_name], errors='coerce')
        # Replace your date column with days since reference date
        self.data[date_as_number_column_name] = (self.data[date_column_name] - reference_date).dt.days


        # Save the 'Date.1' column for later re-insertion
        date_column_data = self.data[date_column_name]

        # Drop columns we factored out, replaced, or that are essentially "cheating" (info not available at game time)

        df_pre_scale = self.data.drop(['Score',
                                      moneyline_target_column_name,
                                       ou_determinant_column_name,
                                      overunder_target_variable_column_name,
                                      date_column_name,
                                      date_1_column_name,
                                      team_name_1_column_name,
                                      team_name_2_column_name], axis=1)

        # Handle missing values
        # imputer = SimpleImputer(strategy='mean')
        # self.X = pd.DataFrame(imputer.fit_transform(self.X), columns=self.X.columns)

        # Normalize numerical features
        scaler = StandardScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(df_pre_scale), columns=df_pre_scale.columns)
        df_scaled[date_column_name] = date_column_data

        self.X = df_scaled


        print("Data preprocessed.")

    def get_raw_data(self):
        return self.raw_data

    def get_processed_data(self):
        return self.data

    def get_feature_data(self):
        return self.X

    def get_label_data(self):
        return self.y

    def split_data(self, test_size=0.2, random_state=42):
        """
        This should move into a DataCut
        :param test_size:
        :param random_state:
        :return:
        """
        print("InitialDataProcessor#split_data")
        # print(self.y.shape) # Should be a one-dimenaional array with shape (2959,)
        #
        # # Ensure self.y is a single-column array or series
        # if isinstance(self.y, pd.DataFrame):
        #     self.y = self.y.iloc[:, 0]  # Select the first column if y is a DataFrame

        return train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)
