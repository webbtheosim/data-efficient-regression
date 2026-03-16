import kmedoids
import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.neighbors import NearestNeighbors, KernelDensity
from sklearn.preprocessing import MinMaxScaler
from sklearn_extra.cluster import KMedoids
from sklearn.metrics.pairwise import (
    pairwise_distances,
    pairwise_distances_argmin,
)
import time

def random(domain, size, seed):
    ''' 
        Select a random sample from the domain. Returns the indices of 
        points chosen from the provided domain.
    '''

    np.random.seed(seed=seed)
    idx = [i for i in range(domain.shape[0])]
    np.random.shuffle(idx)

    return idx[0:size]

def maximin(domain, size, seed, metric='euclidean'):
    '''
        Select a sample that maximizes the minimum distance between any two 
        points in the sample, subject to a randomly chosen initial point.
    '''

    np.random.seed(seed=seed)
    domain = MinMaxScaler().fit_transform(domain)
    sample = [np.random.randint(low=0, high=domain.shape[0])]

    for _ in range(size - 1):

        chosen_points = domain[sample, :]
        distances = cdist(chosen_points, domain, metric=metric)
        distances = np.array(distances)
        distances = np.min(distances, axis=0).reshape(-1,1)
        new_idx = np.argsort(-distances, axis=0)[0].item()
        sample.append(new_idx)

    return sample

def medoids(domain, size, seed, metric='euclidean'):
    '''
        Select a sample that minimizes the variance between all points in the 
        dataset to a selected point. If the dataset is larger than 10,000 points,
        we apply medoids sampling to a randomly chosen subsample of the original
        dataset.
    '''

    if domain.shape[0] < 10000:

        domain = MinMaxScaler().fit_transform(domain)
        diss = squareform(pdist(domain))
        k = kmedoids.fasterpam(
            diss, size, 
            max_iter=100, 
            random_state=seed, 
            n_cpu=1
        )
        sample = k.medoids.tolist()

    else:

        all_idx = np.array([i for i in range(domain.shape[0])])
        np.random.shuffle(all_idx)
        subsampled_idx = all_idx[0:10000]
        subsampled_domain = domain[subsampled_idx]
        subsampled_domain = MinMaxScaler().fit_transform(subsampled_domain)
        diss = squareform(pdist(subsampled_domain))
        k = kmedoids.fasterpam(
            diss, size, 
            max_iter=100, 
            random_state=seed, 
            n_cpu=1
        )
        idx = k.medoids.tolist()
        sample = subsampled_idx[idx].tolist()

    return sample

class FixedKMedoids(KMedoids):
    '''
        Modified version of KMedoids class in scikit-learn-extra that performs k-medoids
        given a fixed set of medoids specified by the user. This is useful for sequential
        k-medoids selection.
    '''

    def __init__(self, n_clusters=10, fixed_idx=None, metric='euclidean',
                method='pam', init='random', max_iter=300, random_state=None):
        super().__init__(n_clusters=n_clusters,
                        metric=metric,
                        method=method,
                        init=init,
                        max_iter=max_iter,
                        random_state=random_state)
        self.fixed_indices = np.array(fixed_idx, dtype=int)

    def fit(self, X, y=None):

        # Compute pairwise distances (as usual)
        X = np.array(X)
        self.X_ = X
        self.D_ = pairwise_distances(X, metric=self.metric)

        n_samples = len(X)
        rng = np.random.default_rng(self.random_state)

        # Initialize medoids: start with fixed ones
        n_fixed = len(self.fixed_indices)
        remaining = np.setdiff1d(np.arange(n_samples), self.fixed_indices)
        n_to_select = self.n_clusters - n_fixed

        if isinstance(self.init, str) and self.init == 'random':
            init_medoids = np.concatenate([
                self.fixed_indices,
                rng.choice(remaining, size=n_to_select, replace=False)
            ])
        elif isinstance(self.init, (list, np.ndarray)):
            init_medoids = np.array(self.init, dtype=int)
        else:
            raise ValueError("init must be 'random' or list of medoid indices")

        medoids = np.sort(init_medoids)

        for _ in range(self.max_iter):
            # Assign each point to closest medoid
            labels = np.argmin(self.D_[:, medoids], axis=1)

            new_medoids = medoids.copy()
            for i, medoid in enumerate(medoids):
                if medoid in self.fixed_indices:
                    continue  # skip fixed medoids
                cluster_points = np.where(labels == i)[0]
                if len(cluster_points) == 0:
                    continue
                intra_dist = self.D_[np.ix_(cluster_points, cluster_points)]
                costs = intra_dist.sum(axis=1)
                new_medoids[i] = cluster_points[np.argmin(costs)]

            if np.all(new_medoids == medoids):
                break
            medoids = new_medoids

        self.medoid_indices_ = np.sort(medoids)
        self.labels_ = np.argmin(self.D_[:, self.medoid_indices_], axis=1)
        return self
def fixed_medoids(domain, chosen_idx, batch_size, seed, metric='euclidean'):
    '''
        Method used for adapative space-filling. Identifies 'batch_size' new medoids
        given a fixed set of 'chosen_idx' medoids which optimizes the k-medoids
        criterion.
    '''
    if domain.shape[0] < 10000:
        idx = [i for i in range(domain.shape[0])]
        np.random.seed(seed)
        np.random.shuffle(idx)
        domain = domain[idx]
        chosen_idx = [idx.index(i) for i in chosen_idx]
        domain = MinMaxScaler().fit_transform(domain)
        kmedoids = FixedKMedoids(
            n_clusters=len(chosen_idx)+batch_size, 
            fixed_idx=chosen_idx, random_state=seed).fit(domain)
        new_idx = [i for i in kmedoids.medoid_indices_ if i not in chosen_idx]
        return [idx[i] for i in new_idx]
    else:
        idx = np.array([i for i in range(domain.shape[0]) if i not in chosen_idx])
        np.random.shuffle(idx)
        subsampled_idx = idx[0:10000 - len(chosen_idx)]
        subsampled_idx = np.hstack((chosen_idx, subsampled_idx))
        chosen_idx = [i for i in range(len(chosen_idx))]
        subsampled_domain = domain[subsampled_idx]
        subsampled_domain = MinMaxScaler().fit_transform(subsampled_domain)
        kmedoids = FixedKMedoids(
            n_clusters=len(chosen_idx)+batch_size, 
            fixed_idx=chosen_idx, random_state=seed).fit(subsampled_domain)
        new_idx = [i for i in kmedoids.medoid_indices_ if i not in chosen_idx]
        return subsampled_idx[new_idx].tolist()

def max_entropy(domain, size, seed, neighbors=1000, metric='euclidean'):
    '''
        Select a sample that maximizes the Shannon entropy of the selection by 
        assuming each point is a Gaussian distribution with bandwidths determined 
        by maximum likelihood estimation. If the dataset is larger than 10,000 points,
        we apply max_entropy sampling to a randomly chosen subsample of the original
        dataset.
    '''

    np.random.seed(seed=seed)
    neighbors = min(neighbors, domain.shape[0])

    all_idx = [i for i in range(domain.shape[0])]
    np.random.shuffle(all_idx)
    subsample_idx = np.array(all_idx[0:min(10000, domain.shape[0])])
    domain = domain[subsample_idx]

    domain = MinMaxScaler().fit_transform(domain)
    sample = [np.random.randint(low=0, high=domain.shape[0])]

    # Compute bandwidths using Scott's rule (per scipy.stats.gaussian_kde) as a default.
    # Ignore those features for which there aren't standard deviations > 0.01.
    stds = np.std(domain, axis=0)
    keep_dimension = np.argwhere(stds > 0.01).reshape(-1)
    domain = domain[:,keep_dimension]
    stds = stds[keep_dimension]
    bandwidths = stds * domain.shape[0] ** -0.2

    # Find neighbors for accelerated overlap calculations.
    nbrs = NearestNeighbors(n_neighbors=neighbors).fit(domain)
    _, neighbor_idx = nbrs.kneighbors(domain)

    def compute_score(input, idx):
        '''
            Method for computing the shared "information" between the point specified
            by "idx" and the set of points specified by "input." Used for parallelizing
            max_entropy calculations. 
        '''

        datum = domain[idx].reshape(1,-1)
        d2 = np.sum(np.square(input - datum) / np.square(bandwidths), axis=1)
        metric = np.sum(np.exp(-d2 / 2.0), axis=0)

        return metric
    
    # Compute an "overlap score" between a point and its neighbors. Represents the density
    # of points near this point, indicating how representative this point is of its neighbors.
    # A more representative point is more likely to be chosen by this method.
    overlap_score = np.array([compute_score(domain[neighbor_idx[i]], i) for i in range(domain.shape[0])])
    overlap_score = overlap_score.reshape(-1,1) / neighbors

    for _ in range(size - 1):

        # Compute a "redundancy score" between a point and the points already chosen. Represents
        # how similar candidate points are to those that have already been chosen. A point that is
        # similar to those already chosen is less likely to be chosen by this method.
        redundant_score = np.array([compute_score(domain[sample,:], i) for i in range(domain.shape[0])])
        redundant_score = redundant_score.reshape(-1,1) / (len(sample) + 1.0)
        total_score = overlap_score - redundant_score

        # Prevent previously chosen points from being chosen again.
        total_score[sample, :] = -np.inf

        new_idx = np.argmax(total_score, axis=0).item()
        sample.append(new_idx)

    sample = subsample_idx[sample].tolist()

    return sample

def vendi_mc(domain, size, seed, neighbors=1000, metric='euclidean', max_time=300):
    '''
        Select sample that maximizes the Vendi score for the chosen selection. 
        The similarity function used in the Vendi score is the same probability
        distribution used for the max_entropy sample selection. Here, we optimize
        the sample for Vendi score using Monte Carlo, since this produces
        reasonable samples faster than greedy selection.

        The optimization terminates after the specified amount of time.
    '''

    np.random.seed(seed=seed)
    neighbors = min(neighbors, domain.shape[0])

    all_idx = [i for i in range(domain.shape[0])]
    np.random.shuffle(all_idx)
    domain = domain[all_idx]
    sample = all_idx[0:size]

    domain = MinMaxScaler().fit_transform(domain)

    # Compute bandwidths using Scott's rule (per scipy.stats.gaussian_kde) as a default.
    # Ignore those features for which there aren't standard deviations > 0.01.
    stds = np.std(domain, axis=0)
    keep_dimension = np.argwhere(stds > 0.01).reshape(-1)
    domain = domain[:,keep_dimension]
    stds = stds[keep_dimension]
    bandwidths = stds * domain.shape[0] ** -0.2

    def compute_vendi(sample):
        ''' Method for computing the Vendi score from the provided indices. '''

        data = domain[sample,:]
        dist = pdist(data, 'seuclidean', V=np.square(bandwidths))
        sim_matrix = squareform(dist)
        sim_matrix = np.exp(-np.square(sim_matrix) / 2.0)
        sim_matrix = sim_matrix / sim_matrix.shape[0]

        eig_vals = np.linalg.eigvals(sim_matrix)
        entropy = -np.sum(np.multiply(eig_vals, np.log(eig_vals + 1e-30)))
        vendi_score = np.exp(entropy)

        return vendi_score
    
    # Get neighbors for viable MC steps.
    nbrs = NearestNeighbors(n_neighbors=neighbors).fit(domain)
    _, neighbor_idx = nbrs.kneighbors(domain)

    # Run MC optimization.
    n_steps = 10000
    temp = 0.1 / size
    curr_score = compute_vendi(sample)
    start = time.perf_counter()
    for trial in range(n_steps):
    
        trial_idx = np.random.randint(low=0, high=len(sample))
        trial_neighbor_idx = np.random.randint(low=1, high=neighbors)
        old_idx = sample[trial_idx]
        new_idx = neighbor_idx[trial_idx, trial_neighbor_idx]

        if new_idx not in sample:
            sample[trial_idx] = new_idx
            new_score = compute_vendi(sample)
            if new_score > curr_score:
                curr_score = new_score
            else:
                trial_prob = np.random.random()
                if trial_prob < np.exp((new_score - curr_score) / temp):
                    curr_score = new_score
                else:
                    sample[trial_idx] = old_idx

            if trial == 0 or (trial + 1) % 100 == 0:
                print(f'Trial: {trial + 1} | Vendi: {curr_score:.3f}')

        if (time.perf_counter() - start) > max_time:
            break

    all_idx = np.array(all_idx)

    return all_idx[sample].tolist()

def sample(name, domain, size, seed, fixed_idx=None):
    ''' Wrapper method for interfacing with the implemented sampling methods. '''

    if name == 'random':
        return random(domain, size, seed)
    elif name == 'maximin':
        return maximin(domain, size, seed)
    elif name == 'medoids':
        return medoids(domain, size, seed)
    elif name == 'fixed-medoids':
        return fixed_medoids(domain, fixed_idx, size, seed)
    elif name == 'max_entropy':
        return max_entropy(domain, size, seed)
    elif name == 'vendi':
        return vendi_mc(domain, size, seed)
    else:
        raise Exception('The sampler name specified is not implemented.')
