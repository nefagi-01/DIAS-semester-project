import time
import numpy as np 

# Helpers
def getLables(X, centroids, get_e=False):
    diff = X[:, None] - centroids[None]  # (n, k, d)
    dist = np.einsum('nkd,nkd->nk', diff, diff)
    labels = dist.argmin(1)
    if get_e:
        dist.sort(1)
        e = dist[:,1] - dist[:,0]
        return labels, e
    return labels

def getCentroids(X, labels, k):
    group_counts = np.bincount(labels, minlength=k)[:, None]
    fn = lambda w: np.bincount(labels, weights=w, minlength=k)
    return np.apply_along_axis(fn, 0, X) / group_counts

def getCorrectedLables(X, centroids, speculated_centroids, e, labels):
    D = np.max(np.nan_to_num(np.linalg.norm(centroids - speculated_centroids, 2, 1)))
    mask_correction = (e - 2*D) <= 0
    if mask_correction.sum() > 0:
        labels[mask_correction] = getLables(X[mask_correction], centroids)
    return labels

    
# Implementations    
def KMeans(X, k, num_iter=50, measure=False):
    n, d = X.shape
    centroids = X[np.random.choice(n, k, replace=False)]  # (k, d)
    if measure:
        A_time = []
        B_time = []
    
    for i in range(num_iter):
        # Save previous labels
        if i > 0:
            prev_labels = labels
        
        if measure:
            start = time.time()
        
        # Assignment step
        labels = getLables(X, centroids)
        
        if measure:
            end = time.time()
            A_time.append(end-start)
            start = time.time()    
            
        # Update step
        centroids = getCentroids(X, labels, k)
        
        if measure:
            end = time.time()
            B_time.append(end-start)
        
        # Check convergence
        if i > 0 and (labels == prev_labels).all():
            if measure:
                return labels, centroids, np.array(A_time), np.array(B_time)
            return labels, centroids
        
    if measure:
        return labels, centroids, np.array(A_time), np.array(B_time)
    return labels, centroids


def KMeans_speculation(X, k, num_iter=50, subsample_size = 0.01, measure=False):    
    # Data definition
    n, d = X.shape
    centroids = X[np.random.choice(n, k, replace=False)]
    # Subsample of points used for speculation
    mask = np.random.choice([True, False], size=X.shape[0], p=[subsample_size, 1-subsample_size])
    
    # Assignment step
    labels = getLables(X, centroids)
    for i in range(num_iter):
        # Save previous labels
        prev_labels = labels

        # Speculate centroids using mask
        # If some clusters are small, they may not be sampled. Therefore, update only centroids whose clusters have subsamples.
        ma = np.ma.masked_invalid(getCentroids(X[mask], labels[mask], k))
        centroids[~ma.mask] = ma.data[~ma.mask]
        
        # Assignment step, using the centroids before speculated
        labels, e = getLables(X, centroids, get_e = True)
                
        # Before update step, save speculated_centroids for correction
        speculated_centroids = centroids
        
        # Update step, using prev_labels (i.e: labels before update due to getLabels done on speculation)
        centroids = getCentroids(X, prev_labels, k)
        
        labels = getCorrectedLables(X, centroids, speculated_centroids, e, labels)
        
        # Check convergence
        if i > 0 and (labels == prev_labels).all():
            return labels, centroids
        
    return labels, centroids