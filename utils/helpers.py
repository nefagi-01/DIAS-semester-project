import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_blobs
from typing import List
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


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
def timeseries_plot(timeseries_list: List[np.ndarray], columns: List[str], xlabel=None, ylabel=None):
    ax = sns.lineplot(data = pd.DataFrame(np.column_stack(timeseries_list), columns=columns))
    if xlabel and ylabel:
        ax.set(xlabel=xlabel, ylabel=ylabel)
    plt.show()

    