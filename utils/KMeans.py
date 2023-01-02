from time import process_time_ns
import numpy as np
import gc
import pandas as pd
from sklearn.cluster import kmeans_plusplus

# Used for converting ns in ms
FACTOR = 1e+06

# Helpers

# Return squared distances between all datapoint, centroid pairs
def getDist(X, centroids):
    diff = X[:, None] - centroids[None]  # (n, k, d)
    return np.einsum('nkd,nkd->nk', diff, diff)

# Return avg squared dist between datapoints and its closest centroid
def getAvgDist(X, centroids):
    return np.mean(getDist(X, centroids).min(1))

def getLables(X, centroids, get_e=False, get_inertia = False):
    dist = getDist(X, centroids)
    labels = dist.argmin(1)
    if get_e:
        # take root of squared distances to have distances
        dist = np.sqrt(dist)
        dist.sort(1)
        e = dist[:,1] - dist[:,0]
        return labels, e
    if get_inertia:
        inertia = np.mean(np.take_along_axis(dist, labels[:,None], axis=1))
        return labels, inertia
    return labels

def getCentroids(X, labels, k, old_centroids):
    centroids = np.copy(old_centroids)
    group_counts = np.bincount(labels, minlength=k)[:, None]
    not_empty_clusters = (group_counts!=0)
    k_not_empty = not_empty_clusters.sum()
    fn = lambda w: np.bincount(labels, weights=w, minlength=k)
    centroids[not_empty_clusters.reshape(-1)] = np.apply_along_axis(fn, 0, X)[not_empty_clusters.reshape(-1)] / group_counts[not_empty_clusters, None]
    return centroids

def getCorrectedLables(X, centroids, speculated_centroids, e, labels):
    D = np.max(np.nan_to_num(np.linalg.norm(centroids - speculated_centroids, 2, 1)))
    mask_correction = (e - 2*D) <= 0
    if mask_correction.sum() > 0:
        labels[mask_correction] = getLables(X[mask_correction], centroids)
    return labels

    
# Implementations    
def KMeans(X, k, num_iter=50, seed = 0, measure=False, kmeans_pp = False, tol = 1e-6, return_steps = False, centroids = None, measure_inertia = False):
    n, d = X.shape
    
    if centroids is None:
        if kmeans_pp:
            # Calculate seeds from kmeans++
            centroids, _ = kmeans_plusplus(X, n_clusters=k, random_state=seed)
        else:
            np.random.seed(seed)
            centroids = X[np.random.choice(n, k, replace=False)]  # (k, d)
            
    if measure:
        A_time = []
        B_time = []
        # Disable gc to have more precise measurements
        gc.disable()
    if measure_inertia:
        inertia_list = []
    
    precomputed = False
    
    for i in range(num_iter):
        # Save previous labels
        if i > 0:
            prev_labels = labels
        
        # Assignment step
        if not precomputed:
            labels = getLables(X, centroids)
        
        if measure:
            end = process_time_ns()
            A_time.append((end-start)/FACTOR)
            start = process_time_ns()    
            
        # Update step
        centroids = getCentroids(X, labels, k, centroids)
        
        if measure:
            end = process_time_ns()
            B_time.append((end-start)/FACTOR)
        
        if i > 0:
            prev_inertia = inertia
            
        labels, inertia = getLables(X, centroids, get_inertia = True)
        precomputed = True
        
        if measure_inertia:
            inertia_list.append(inertia)
        
        # Check convergence - use relative difference
        if i > 0 and ((labels == prev_labels).all() or np.abs((inertia-prev_inertia)/inertia) <= tol):
            if measure:
                # Re-enable gc
                gc.enable()
                return labels, centroids, np.array(A_time), np.array(B_time)
            if return_steps:
                return labels, centroids, i
            if measure_inertia:
                return labels, centroids, inertia_list
            return labels, centroids
        
    if measure:
        # Re-enable gc
        gc.enable()
        return labels, centroids, np.array(A_time), np.array(B_time)
    
    if measure_inertia:
        return labels, centroids, inertia_list
    
    if return_steps:
        return labels, centroids, i
    
    return labels, centroids


def KMeans_speculation(X, k, num_iter=50, subsample_size = 0.01, measure=False):    
    # Data definition
    n, d = X.shape
    centroids = X[np.random.choice(n, k, replace=False)]
    # Subsample of points used for speculation
    mask = np.random.choice([True, False], size=X.shape[0], p=[subsample_size, 1-subsample_size])
    
    if measure:
        A_time = []
        B_time = []
        speculation_time = []
        correction_time = []
        # Disable gc to have more precise measurements
        gc.disable()
    
    # Assignment step
    labels = getLables(X, centroids)
    for i in range(num_iter):
        # Save previous labels
        prev_labels = labels

        if measure:
            start = process_time_ns()
        # Speculate centroids using mask
        # If some clusters are small, they may not be sampled. Therefore, update only centroids whose clusters have subsamples.
        ma = np.ma.masked_invalid(getCentroids(X[mask], labels[mask], k))
        centroids[~ma.mask] = ma.data[~ma.mask]
        if measure:
            end = process_time_ns()
            speculation_time.append((end-start)/FACTOR)
            start = process_time_ns()
        
        # Assignment step, using the centroids before speculated
        labels, e = getLables(X, centroids, get_e = True)
        if measure:
            end = process_time_ns()
            A_time.append((end-start)/FACTOR)
                
        # Before update step, save speculated_centroids for correction
        speculated_centroids = centroids
        
        if measure:
            start = process_time_ns()
        # Update step, using prev_labels (i.e: labels before update due to getLabels done on speculation)
        centroids = getCentroids(X, prev_labels, k)
        if measure:
            end = process_time_ns()
            B_time.append((end-start)/FACTOR)
            start = process_time_ns()
            
        labels = getCorrectedLables(X, centroids, speculated_centroids, e, labels)
        
        if measure:
            end = process_time_ns()
            correction_time.append((end-start)/FACTOR)
        
        # Check convergence
        if i > 0 and (labels == prev_labels).all():
            if measure:
                # Re-enable gc
                gc.enable()
                return labels, centroids, A_time, B_time, speculation_time, correction_time
            return labels, centroids
        
    if measure:
        # Re-enable gc
        gc.enable()
        return labels, centroids, A_time, B_time, speculation_time, correction_time
    
    return labels, centroids


def subsample(vector, sample_size, offset):
    return vector[offset : offset + sample_size], offset + sample_size
    

def KMeans_sketching(X, k, num_iter=50, seed=None, subsample_size = 0.01, save = False, path='./data.csv', measure = False, choose_best = False, resampling = False, trace=False, tol = 1e-3, return_steps = False, measure_time = False, resample_centroid = False, tol_resampling_centroids = 1e-2, MAX_COUNTER = 3, kmeans_pp = False, hybrid = False, p = 0.5, centroids = None, q = 0.5):
    
    if measure:
        # list of L_diff
        L_diff_list = []
        L_slow_list = []
        L_fast_list = []
        
    if measure_time:
        A_time = []
        B_time = []
        sampling_time = []
        sampling_centroids_time = []
        choose_best_time = []
        sketching_time = []
        getAvg_time = []
        permutation_time = []
        # Disable gc to have more precise measurements
        gc.disable()
    
    
    n, d = X.shape
    
    if measure_time:
        start = process_time_ns()
            
    # permute X, this will be used for random sampling of both datapoints and centroids
    np.random.seed(seed)
    X_perm_k = np.random.permutation(X)
    np.random.seed(seed + 1)
    X_perm_subsample = np.random.permutation(X)
    
    # define offsets used for accessing the permutation with a sliding window
    offset_k = 0
    offset_X = 0
    size_X_subsample = int(np.ceil(subsample_size * n))
    
    if centroids is  None:
        if kmeans_pp:
            # Calculate seeds from kmeans++
            centroids, _ = kmeans_plusplus(X, n_clusters=k, random_state=seed)
        else:
            # subsample centroids
            centroids, offset_k = subsample(X_perm_k, k, offset_k)  # (k, d)
        
    # subsample datapoints
    X_subsample, offset_X = subsample(X_perm_subsample, size_X_subsample, offset_X)

    if measure_time:
        end = process_time_ns()
        permutation_time.append((end-start)/FACTOR)

       
    # Number of executions n_executions = 1/subsample_size
    n_executions = int(np.floor(1/subsample_size))
    
    # Counter to decide when to switch off resample centroids
    counter = 0
    # Trace counter 
    trace_counter = 1
    
    
        
    precomputed = False
        
                
    for i in range(num_iter):
        # Save previous labels and L_slow
        if i > 0:
            prev_labels = labels
            prev_L_slow = L_slow
            if measure_time:
                permutation_time.append(0)
            
        fast_centroids = centroids    
        
        # SLOW EXECUTION
        
        if measure_time:
            start = process_time_ns()
            
        if not precomputed:
            # Assignment step
            labels = getLables(X, centroids)
        
        if measure_time:
            end = process_time_ns()
            A_time.append((end-start)/FACTOR)
            start = process_time_ns()

        # Update step
        centroids = getCentroids(X, labels, k, centroids)
        
        if measure_time:
            end = process_time_ns()
            B_time.append((end-start)/FACTOR)
        
        
        
        # FAST EXECUTION
        
        
        if measure_time:
            start = process_time_ns()
        # Resample centroids
        if resample_centroid:
            # subsample centroids
            resampled_centroid, offset_k = subsample(X_perm_k, k, offset_k)  # (k, d)
            # add randomness to centroids
            fast_centroids = p*fast_centroids + (1-p)*resampled_centroid
        if measure_time:
            end = process_time_ns()
            sampling_centroids_time.append((end-start)/FACTOR)
                
        if measure_time:
            start = process_time_ns()
            
        # Execute (a,b) n_execution times
        for j in range(n_executions):
            # A - Assignment step
            fast_labels = getLables(X_subsample, fast_centroids)
            # B - Update step
            fast_centroids = getCentroids(X_subsample, fast_labels, k, fast_centroids)
            
        if measure_time:
            end = process_time_ns()
            sketching_time.append((end-start)/FACTOR)
        
        if resampling and trace:
            if i > 0:
                # fast_centroids = (1/trace_counter) * fast_centroids + (1-1/trace_counter) * old_centroids
                # trace_counter += 1
                fast_centroids = q * fast_centroids + (1-q) * old_centroids
            

        # Compute avg distance
        if measure_time:
            start = process_time_ns()
        labels, L_slow = getLables(X, centroids, get_inertia = True)   
        if i > 0:
            prev_L_fast = L_fast
        fast_labels, L_fast = getLables(X, fast_centroids, get_inertia = True)
        precomputed = True
        if measure_time:
            end = process_time_ns()
            getAvg_time.append((end-start)/FACTOR)
        if measure:
            
            L_slow_list.append(L_slow)
            L_fast_list.append(L_fast)

            # L_diff
            L_diff_list.append(L_fast - L_slow)
            
        # Choose best centroids between fast and slow
        if choose_best:
            if measure_time:
                start = process_time_ns()
                
            if L_fast < L_slow:
                labels = fast_labels
                centroids = fast_centroids
                L_slow = L_fast
                
            if measure_time:
                end = process_time_ns()
                choose_best_time.append((end-start)/FACTOR)
                
        if resampling and trace:
            old_centroids = centroids
                
        # Resample datapoints
        if resampling:
            if measure_time:
                start = process_time_ns()
            # subsample datapoints
            X_subsample, offset_X = subsample(X_perm_subsample, size_X_subsample, offset_X)
            if measure_time:
                end = process_time_ns()
                sampling_time.append((end-start)/FACTOR)
                
        if hybrid:    
            # check if fast_centroids leads to similar result of previous fast_centroids
            if i > 0 and resample_centroid and np.abs(prev_L_fast - L_fast)/L_fast <= tol_resampling_centroids:
                counter += 1
            else:
                counter = 0

            # if we keep gettin similar vector with centroid resampling -> we found optimum position
            if resample_centroid and counter >= MAX_COUNTER:
                resample_centroid = False
                trace = True
        
        # Check convergence - use relative difference
        if i > 0 and ((labels == prev_labels).all() or np.abs((L_slow-prev_L_slow)/L_slow) <= tol):
            break
        
            
    # Save data to file
    if measure and save:
        df = pd.DataFrame(np.column_stack([L_slow_list, L_fast_list, L_diff_list]), columns=['L_slow', 'L_fast', 'L_diff'])
        df['n'] = n
        df['d'] = d
        df['k'] = k
        df['seed'] = seed
        df['subsample_size'] = subsample_size
        df['steps'] = i
        if measure_time:
            df['t_A'] = A_time
            df['t_B'] = B_time
            df['t_sketching'] = sketching_time
            df['t_sampling'] = sampling_time
            df['t_choose_best'] = choose_best_time
            if len(sampling_centroids_time) > 0:
                df['t_sampling_centroids'] = sampling_centroids_time
            df['t_get_avg'] = getAvg_time
            df['t_permutation'] = permutation_time
            gc.enable()
        df.to_csv(path,index=False)
    
    if return_steps:
        return labels, centroids, i
    
    return labels, centroids
