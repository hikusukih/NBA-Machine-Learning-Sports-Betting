import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import pandas as pd
import numpy as np

# Connect to the SQLite database

con_ds = sqlite3.connect("../../Data/dataset.sqlite")
con_od = sqlite3.connect("../../Data/odds.sqlite")
con_te = sqlite3.connect("../../Data/teams.sqlite")

conn = []
conn.append(con_ds)
# conn.append(con_od)
# conn.append(con_te)
for con in conn:
    # Load data into a DataFrame
    dataset = "dataset_2012-24"
    query = f"SELECT * FROM \"{dataset}\" Order By Date"
    data = pd.read_sql_query(query, con)

    numeric_data = data.select_dtypes(include=[np.number])

    # Display the first few rows of the dataframe
    # print(data.head())

    # Get summary statistics and info about the dataset
    # print(data.describe())
    # print(data.info())

    # Plotting histograms for all numeric features
    # data.hist(figsize=(20, 15))
    # plt.show()

    # # Boxplot for a specific feature
    # data.boxplot(column=['FeatureName'])
    # plt.show()


    # Pairplot for a subset of features
    # sns.pairplot(data[['W', 'L', 'W_PCT', 'MIN']])
    # plt.show()

    # Correlation matrix
    corr_matrix = numeric_data.corr()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f")
    plt.show()

    # Plotting the distribution to check for normalization
    for column in data.columns:
        data[column].hist()
        plt.title(column)
        plt.show()
