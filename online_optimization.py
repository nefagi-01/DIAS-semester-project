from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import queue
from multiprocessing import Process
import random
import time



def generate_clusters(n_clusters=3, d=2, n=100):
    centers = np.random.rand(n_clusters, d) * 15
    cluster_std = np.random.normal(1, 0.2, n_clusters)
    X, y = make_blobs(n_samples=n, cluster_std=cluster_std, centers=centers, n_features=d, random_state=1)

    # print("2d plot")
    # for cluster in range(n_clusters):
    #     plt.scatter(X[y == cluster, 0], X[y == cluster, 1], s=10, label=f"Cluster{cluster}")
    return centers, cluster_std, X, y


# assign cluster to datapoint
def A(x, centers):
    distances = np.linalg.norm(centers - x, 2, 1)
    return int(np.argmin(distances))


def update_center(x, center, count, add=True):
    if add:
        center = (center * count + x) / (count + 1)
        return center, count + 1
    else:
        if (count - 1) > 0:
            center = (center * count - x) / (count - 1)
        return center, max(0, count - 1)


# update centers
def B(cluster_count, x, centers, id_cluster, previous_assignment, i):
    if id_cluster != previous_assignment[i]:
        # if not first assignment, then update center of previous cluster by removing x
        if previous_assignment[i] != -1:
            centers[previous_assignment[i]], cluster_count[previous_assignment[i]] = update_center(x, centers[
                previous_assignment[i]], cluster_count[previous_assignment[i]], add=False)
        # update center of new cluster assigned to
        centers[id_cluster], cluster_count[id_cluster] = update_center(x, centers[id_cluster],
                                                                       cluster_count[id_cluster], add=True)
        previous_assignment[i] = id_cluster
    return cluster_count, centers, previous_assignment


def k_means(X, n_clusters, n_iterations, d):
    centers = np.random.rand(n_clusters, d) * 15
    cluster_count = np.zeros(n_clusters)
    previous_assignment = np.ones(n).astype(int) * (-1)
    for j in range(n_iterations):
        for i, x in enumerate(X):
            id_cluster = A(x, centers)
            cluster_count, centers, previous_assignment = B(cluster_count, x, centers, id_cluster, previous_assignment,
                                                            i)
    return cluster_count, centers, previous_assignment



def k_means_optimized(X, n_clusters, n_iterations, d):
    n = len(X)
    centres = np.random.rand(n_clusters, d) * 15
    cluster_count = np.zeros(n_clusters)
    previous_assignment = np.ones(n).astype(int) * (-1)
    queue_A = queue.Queue()
    [queue_A.put((i, x)) for i, x in enumerate(X)]
    queue_B = queue.Queue()
    def B_consumer():
        for j in range(n * n_iterations):
            i, x, id_cluster = queue_B.get()
            if id_cluster != previous_assignment[i]:
                # if not first assignment, then update center of previous cluster by removing x
                if previous_assignment[i] != -1:
                    centres[previous_assignment[i]], cluster_count[previous_assignment[i]] = update_center(x, centres[
                        previous_assignment[i]], cluster_count[previous_assignment[i]], add=False)
                    cluster_count[previous_assignment[i]] -= 1
                # update center of new cluster assigned to
                centres[id_cluster], cluster_count[id_cluster] = update_center(x, centres[id_cluster],
                                                                               cluster_count[id_cluster], add=True)
                previous_assignment[i] = id_cluster
            queue_A.put((i, x))

    def A_producer():
        for j in range(n * n_iterations):
            # .get() is a blocking operation
            i, x = queue_A.get()
            id_cluster = A(x, centres)
            queue_B.put((i, x, id_cluster))

    # create a producer process
    producer_process = Process(target=A_producer)
    producer_process.start()

    # create a consumer process
    consumer_process = Process(target=B_consumer)
    consumer_process.start()

    # wait for all processes to finish
    producer_process.join()
    consumer_process.join()

    return cluster_count, centres, previous_assignment


n_clusters = 4
d = 2
n = 200
n_iterations = 1000
centers, cluster_std, X, y = generate_clusters(n_clusters, d, n)

start_time = time.time()
cluster_count, result, previous_assignment  = k_means(X, n_clusters, n_iterations, d)
print("--- %s seconds k_means ---" % (time.time() - start_time))

start_time = time.time()
cluster_count, centers, previous_assignment = k_means_optimized(X, n_clusters, n_iterations, d)
print("--- %s seconds k_means_optimized ---" % (time.time() - start_time))

