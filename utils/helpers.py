import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_blobs
from typing import List
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from openml.datasets import list_datasets, get_datasets



def fit_linear_regression(df, x_name, y_name):
    X = df[x_name].to_numpy().reshape(-1, 1)  
    y = df[y_name].to_numpy()

    model = LinearRegression()
    model.fit(X, y);
    print(f'Model with x: {x_name}, y: {y_name}')
    print(f'\tCoefficient: {model.coef_}')
    print(f'\tIntercept: {model.intercept_}')
    return model


def clean_dataset(df):
    # Remove inf values
    df_clean = df.replace([np.inf, -np.inf], np.nan)
    df_clean = df_clean.dropna()
    return df_clean


def generate_clusters(n_clusters=3, d=2, n=100, seed = None):
    if seed is not None:
        np.random.seed(seed)
    centers = np.random.rand(n_clusters, d) * 15
    cluster_std = np.random.normal(1, 0.2, n_clusters)
    X, y = make_blobs(n_samples=n, cluster_std=cluster_std, centers=centers, n_features=d, random_state=1)

    # print("2d plot")
    # for cluster in range(n_clusters):
    #     plt.scatter(X[y == cluster, 0], X[y == cluster, 1], s=10, label=f"Cluster{cluster}")
    return centers, cluster_std, X, y

# function for plotting list of timeseries
def timeseries_plot(df, xlabel: str = None, ylabel: str = None, show: bool = True, ax = None):
    ax = sns.lineplot(data = df, ax = ax)
    if xlabel and ylabel:
        ax.set(xlabel=xlabel, ylabel=ylabel)
    if show:
        plt.show()
    return ax



"""
Datasets are retrieved from OpenML using an API. It consists in the following steps:
1. In the query we specify the characteristics of the datasets we want to rietrieve.
2. We obtain the ids of these datasets and save it in dataset_ids.csv
3. We load the datasets with the given ids using get_datasets()
4. We transform them in numpy
"""
def load_datasets(query, n_datasets = 10, search = True):
    if search:
        # Search datasets from OpenML given these characteristics
        query = query

        dataset_dataframe = list_datasets(output_format="dataframe").query(query)
        dataset_dataframe = dataset_dataframe.drop_duplicates(['name']).drop_duplicates(['NumberOfNumericFeatures']).sort_values(by=['NumberOfNumericFeatures'])
        dataset_ids = dataset_dataframe['did'][:n_datasets]
        # Save datasets ids into .csv file
        dataset_ids.to_csv('./data/dataset_ids.csv', index=False)
    
    # Load datasets    
    dataset_ids = pd.read_csv('./data/dataset_ids.csv')['did']
    dataset_list = get_datasets(dataset_ids=dataset_ids)
    X_list = [dataset.get_data()[0].select_dtypes([np.number]).to_numpy() for dataset in dataset_list]
    return X_list

    