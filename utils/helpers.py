import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_blobs
from typing import List
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
# from openml.datasets import list_datasets, get_datasets
from openml.datasets import list_datasets
import openml
openml.datasets.functions._get_dataset_parquet = lambda x: None 
from sklearn import cluster, datasets, mixture
from sklearn.cluster import KMeans
from matplotlib.ticker import MaxNLocator
from os import listdir
from os.path import isfile, join
import arff
from utils.KMeans import getAvgDist
from utils.KMeans import KMeans as KMeans_personal



def extend_df(df, m):
    n = df.shape[0]
    if m > n:
        df = df.append([df.iloc[-1]] * (m - n), ignore_index=True)
    return df

def extend_array(array, m):
    if len(array) >= m:
        return array
    else:
        return extend_array(array, m-1) + [array[-1]]

def clean_dataset(df):
    # Remove inf values
    df_clean = df.replace([np.inf, -np.inf], np.nan)
    df_clean = df_clean.dropna()
    return df_clean

def find_bad_initalization(X, k, tol = 1e-6, seed = 0, iteration = 10, max_iter = 50):
    optimal_centroid = estimate_optimal_centroids(X, k, seed = seed, tol = tol)
    optimal_inertia = getAvgDist(X, optimal_centroid)
    worst_centroids = optimal_centroid
    worst_inertia = optimal_inertia
    n, d = X.shape
    for i in range(iteration):
        centroids = X[np.random.choice(n, k, replace=False)]  # (k, d)
        _, centroids = KMeans_personal(X, k, num_iter=max_iter, seed = seed, tol = tol, centroids = centroids)
        inertia = getAvgDist(X, centroids)
        if inertia > worst_inertia:
            worst_inertia = inertia
            worst_centroids = centroids
    return worst_centroids

def fit_linear_regression(df, x_name, y_name, print_text = True):
    X = df[x_name].to_numpy().reshape(-1, 1)  
    y = df[y_name].to_numpy()

    model = LinearRegression()
    model.fit(X, y);
    if print_text:
        print(f'Model with x: {x_name}, y: {y_name}')
        print(f'\tCoefficient: {model.coef_}')
        print(f'\tIntercept: {model.intercept_}')
    return model

def estimate_optimal_centroids(X, k, seed = 0,n_init = 50, tol = 1e-6):
    kmeans = KMeans(n_clusters = k, random_state = seed , n_init = n_init, tol = tol).fit(X)
    centroids = kmeans.cluster_centers_
    return centroids

def clean_dataset(df):
    # Remove inf values
    df_clean = df.replace([np.inf, -np.inf], np.nan)
    df_clean = df_clean.dropna()
    return df_clean


def generate_clusters(n_clusters=3, d=2, n=100, seed = None, plot = False):
    if seed is not None:
        np.random.seed(seed)
    centers = np.random.rand(n_clusters, d) * 15
    cluster_std = np.random.normal(1, 0.2, n_clusters)
    X, y = make_blobs(n_samples=n, cluster_std=cluster_std, centers=centers, n_features=d, random_state=1)

    if plot:
        print("2d plot")
        for cluster in range(n_clusters):
            plt.scatter(X[y == cluster, 0], X[y == cluster, 1], s=10, label=f"Cluster{cluster}")
    return centers, cluster_std, X, y

# function for plotting list of timeseries
def timeseries_plot(df, xlabel: str = None, ylabel: str = None, show: bool = True, ax = None, title = None):
    df.index = df.index + 1
    ax = sns.lineplot(data = df, ax = ax)
    # enforce integer ticks on x-axis
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if xlabel and ylabel:
        ax.set(xlabel=xlabel, ylabel=ylabel)
    if title:
        ax.set_title(title)
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
def load_datasets(query, n_datasets = 10, search = True, return_meta = False):
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
    dataset_list =[openml.datasets.get_dataset(id) for id in dataset_ids]
    X_list = [dataset.get_data()[0].select_dtypes([np.number]).to_numpy() for dataset in dataset_list]
    if return_meta:
        return X_list, dataset_list
    return X_list


def generate_complex_datasets(n_samples, seed):
    # set seed
    np.random.seed(seed)
    
    # ============
    # Generate datasets. We choose the size big enough to see the scalability
    # of the algorithms, but not too big to avoid too long running times
    # ============
    n_samples = n_samples
    noisy_circles = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05)
    noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0.05)
    blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
    no_structure = np.random.rand(n_samples, 2), None

    # Anisotropicly distributed data
    random_state = seed
    X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state, center_box = (-40, 40))
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    aniso = (X_aniso, y)

    # blobs with varied variances
    varied = datasets.make_blobs(
        n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state, center_box = (-40, 40)
    )
    
    default_base = {
    "quantile": 0.3,
    "eps": 0.3,
    "damping": 0.9,
    "preference": -200,
    "n_neighbors": 3,
    "n_clusters": 3,
    "min_samples": 7,
    "xi": 0.05,
    "min_cluster_size": 0.1,
    }

    dataset_list = [
        {
            "name": "Noisy circles",
            "dataset": noisy_circles,
             "parameters":   {
                    "damping": 0.77,
                    "preference": -240,
                    "quantile": 0.2,
                    "n_clusters": 2,
                    "min_samples": 7,
                    "xi": 0.08,
                }
        },
        {
            "name": "Noisy moons",
            "dataset": noisy_moons,
            "parameters": {
                "damping": 0.75,
                "preference": -220,
                "n_clusters": 2,
                "min_samples": 7,
                "xi": 0.1,
            },
        },
        {
            "name": "Varied",
            "dataset": varied,
            "parameters": {
                "eps": 0.18,
                "n_neighbors": 2,
                "min_samples": 7,
                "xi": 0.01,
                "min_cluster_size": 0.2,
            },
        },
        {
            "name": "Anisotropic",
            "dataset": aniso,
            "parameters": {
                "eps": 0.15,
                "n_neighbors": 2,
                "min_samples": 7,
                "xi": 0.1,
                "min_cluster_size": 0.2,
            },
        },
        {
            "name": "Blobs",
             "dataset": blobs,
             "parameters": {
                 "min_samples": 7,
                 "xi": 0.1,
                 "min_cluster_size": 0.2}
        },
        {
            "name": "No structure",
            "dataset": no_structure,
            "parameters": {}
        },
    ]
    
    return dataset_list
    
    

"""
Import datasets manually picked and downloaded in the folder /datasets/
Function used when OpenML servers do not respond to queries, when using function `load_datasets`
"""
def load_downloaded_datasets(path = './datasets/'):
    files = [f for f in listdir(path) if isfile(join(path, f))]

    X_list = []

    for file in files:
        dataset = arff.load(open(path + file, 'r'))
        data = np.array(dataset['data'])
        index_numerical = [ i for i, attribute in enumerate(dataset['attributes']) if attribute[1] == 'REAL' or attribute[1] == 'INTEGER' or attribute[1] == 'NUMERIC']
        data = data[:, index_numerical]
        data = data.astype(np.float64)
        X_list.append(data)
        
    return X_list


def agg_and_plot(df, x, y, ax = None):
    df_agg = df.groupby(x).agg({y : np.mean}).reset_index()
    sns.lineplot(data=df_agg, x=x, y=y, ax = ax)