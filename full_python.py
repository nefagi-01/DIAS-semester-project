# This is a sample Python script.

# Press Maiusc+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from trino.dbapi import connect
import numpy as np
import random

# Connect to Trino instance
conn = connect(
    host="localhost",
    port=8080,
    user="admin",
    catalog="tpch",
    schema="tiny",
)


def cluster_points(X, mu):
    clusters = {}
    for x in X:
        bestmukey = min([(i[0], np.linalg.norm(x - mu[i[0]])) \
                         for i in enumerate(mu)], key=lambda t: t[1])[0]
        try:
            clusters[bestmukey].append(x)
        except KeyError:
            clusters[bestmukey] = [x]
    return clusters


def reevaluate_centers(mu, clusters):
    newmu = []
    keys = sorted(clusters.keys())
    for k in keys:
        newmu.append(np.mean(clusters[k], axis=0))
    return newmu


def has_converged(mu, oldmu):
    # if no centroid has changed => it has converged
    return (set([tuple(a) for a in mu]) == set([tuple(a) for a in oldmu]))


def find_centers(X, K):
    # Initialize to K random centers
    oldmu = random.sample(X, K)
    mu = random.sample(X, K)
    counter = 0
    while not has_converged(mu, oldmu):
        print(f'Iteration {counter}')
        if counter > 15:
            return (mu, clusters)
        oldmu = mu
        # Assign all points in X to clusters
        clusters = cluster_points(X, mu)
        # Reevaluate centers
        mu = reevaluate_centers(oldmu, clusters)
        counter += 1
    return (mu, clusters)


numerical_columns = "orderkey, partkey, suppkey, linenumber, quantity, extendedprice, discount, tax"
cur = conn.cursor()
cur.execute(f"SELECT {numerical_columns} FROM tpch.tiny.lineitem")
X = cur.fetchall()
X = [np.array(x) for x in X]
k = 10

mu, clusters = find_centers(X, k)

print(mu)
