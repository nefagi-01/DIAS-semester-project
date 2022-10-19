import time
import numpy as np 

class KMeans:
    
    def __init__(self, X, n_clusters=4):
        self.X = X
        self.K = n_clusters
        self.centroids = X[np.random.choice(len(X), self.K, replace=False)]
        self.intial_centroids = self.centroids
        self.prev_label,  self.labels = None, np.zeros(len(X))
        
    def fit(self):
        while not np.all(self.labels == self.prev_label) :
            self.prev_label = self.labels
            self.labels = self.predict(self.X)
            self.update_centroid(self.X)
        return self
        
    def predict(self, X):
        return np.apply_along_axis(self.compute_label, 1, X)

    def compute_label(self, x):
        return np.argmin(np.sqrt(np.sum((self.centroids - x)**2, axis=1)))

    def update_centroid(self, X):
        self.centroids = np.array([np.mean(X[self.labels == k], axis=0)  for k in range(self.K)])
        
        
class KMeans_fast: 
    def __init__(self, X, n_clusters=4):
        self.X = X
        self.K = n_clusters
        self.N, self.D = X.shape
        self.centroids = X[np.random.choice(len(X), self.K, replace=False)]
        self.intial_centroids = self.centroids
        self.prev_label,  self.labels = None, np.zeros(len(X))
        # Expansion of numpy arrays in order to have only matrix operations for the assignment. No for loops
        self.X_expanded = np.repeat(X[...,None],self.K,axis=2)
        self.centroids_expanded =  np.repeat(self.centroids.T.reshape(1, self.D, self.K)[...,None], self.N,axis=0).reshape(self.X_expanded.shape)
        
    def fit(self, measure=False):
        A_time = []
        B_time = []
        while not np.all(self.labels == self.prev_label) :
            self.prev_label = self.labels
            if measure:
                start = time.time()
            self.labels = self.predict()
            if measure:
                end = time.time()
                A_time.append(end-start)
                start = time.time()
            self.update_centroid()
            if measure:
                end = time.time()
                B_time.append(end-start)
        if measure:
            return np.array(A_time), np.array(B_time)
        return
    
    def getClustersSize(self):
        return np.array([self.X[self.labels == k].shape[0] for k in range(self.K)])
        
    def predict(self):
        return np.argmin(np.linalg.norm(self.X_expanded - self.centroids_expanded, 2, 1), 1)

    def update_centroid(self):
        self.centroids = np.array([np.mean(self.X[self.labels == k], axis=0)  for k in range(self.K)])
        # Expansion of numpy arrays in order to have only matrix operations for the assignment. No for loops
        self.centroids_expanded =  np.repeat(self.centroids.T.reshape(1, self.D, self.K)[...,None],self.N,axis=0).reshape(self.X_expanded.shape)
        
        
def KMeans_fast_refactor(X, k, num_iter=50, measure=False):
    
    def getLables(X, centroids):
        diff = X[:, None] - centroids[None]  # (n, k, d)
        return np.einsum('nkd,nkd->nk', diff, diff).argmin(1)
    
    def getCentroids(X, labels, k):
        group_counts = np.bincount(labels, minlength=k)[:, None]
        fn = lambda w: np.bincount(labels, weights=w, minlength=k)
        return np.apply_along_axis(fn, 0, X) / group_counts
    
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


                
        
class KMeans_Speculation:
    
    def __init__(self, X, n_clusters=4, n_bins=5, subsample_size = 0.1, max_iterations = 1000):
        self.X = X
        self.K = n_clusters
        self.N, self.D = X.shape
        self.bins = np.linspace(0,30,n_bins)
        self.centroids = X[np.random.choice(len(X), self.K, replace=False)]
        self.speculated_centroids = X[np.random.choice(len(X), self.K, replace=False)]
        self.prev_label,  self.labels = None, np.zeros(len(X))
        self.subsample_size = subsample_size
        # Expansion of numpy arrays in order to have only matrix operations for the assignment. No for loops
        self.X_expanded = np.repeat(X[...,None],self.K,axis=2)
        self.centroids_expanded =  np.repeat(self.centroids.T.reshape(1, self.D, self.K)[...,None], self.N,axis=0).reshape(self.X_expanded.shape)
        self.speculated_centroids_expanded =  np.repeat(self.speculated_centroids.T.reshape(1, self.D, self.K)[...,None], self.N,axis=0).reshape(self.X_expanded.shape)
        self.max_iterations = max_iterations

    def fit(self, measure=False):
        A_time = []
        B_time = []
        speculation_time = []
        correction_time = []
        # First assignment necessary to have speculation of centroids afterwards
        self.labels = self.predict(self.X_expanded, self.centroids_expanded)
        # Subsample of datapoints
        self.mask = np.random.choice([True, False], size=self.X.shape[0], p=[self.subsample_size, 1-self.subsample_size])
        counter = 0
        while (not (self.labels == self.prev_label).all()) and counter < self.max_iterations:
            self.prev_label = self.labels
            if measure:
                start = time.time()
            self.speculate_centroid(self.X)
            if measure:
                end = time.time()
                speculation_time.append(end-start)
                start = time.time()
            self.labels, self.e = self.predict(self.X_expanded, self.centroids_expanded, compute_e = True)
            if measure:
                end = time.time()
                A_time.append(end-start)
            # Save speculated centroids before updating it with correct value.
            self.speculated_centroids = self.centroids
            self.speculated_centroids_expanded = self.centroids_expanded
            # Update centroids using all dataset
            if measure:
                start = time.time()
            self.update_centroid(self.X, self.prev_label)
            if measure:
                end = time.time()
                B_time.append(end - start)
                start = time.time()
            self.correct()
            if measure:
                end = time.time()
                correction_time.append(end-start)
            counter += 1
        if measure:
            return A_time, B_time, speculation_time, correction_time
            
        return self
    
    def speculate_centroid(self, X):
        self.update_centroid(X[self.mask], self.labels[self.mask])
        
    def predict(self, X_expanded, centroids_expanded, compute_e = False):
        distances = np.linalg.norm(X_expanded - centroids_expanded, 2, 1)
        labels = np.argmin(distances, 1)
        if compute_e:
            distances.sort(1)
            e = distances[:,1] - distances[:,0]
            return labels, e
        return labels
    
    def correct(self):
        digitized = np.digitize(self.e, self.bins)
        D = np.max(np.nan_to_num(np.linalg.norm(self.centroids - self.speculated_centroids, 2, 1)))
        mask = (self.e - 2*D) <= 0
        #print(f'Number of corrected points: {mask.sum()}')
        if mask.sum() > 0:
            self.labels[mask] = self.predict(self.X_expanded[mask], self.centroids_expanded[mask])        

    def update_centroid(self, X, labels):
        self.centroids = np.array([np.mean(X[labels == k], axis=0)  for k in range(self.K)])
        # Expansion of numpy arrays in order to have only matrix operations for the assignment. No for loops
        self.centroids_expanded =  np.repeat(self.centroids.T.reshape(1, self.D, self.K)[...,None],self.N,axis=0).reshape(self.X_expanded.shape)
        
    def getClustersSize(self):
        return np.array([self.X[self.labels == k].shape[0] for k in range(self.K)])
        
