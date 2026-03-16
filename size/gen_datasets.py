import kmedoids
import numpy as np
from scipy.spatial.distance import squareform, pdist
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from utils import load_data
from models import *
from batch_selection import select_batch
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin_min

import functools
print = functools.partial(print, flush=True)

def large_scale_kmedoids(
    X, n_clusters=100, batch_size=10000, 
    random_state=0, verbose=0
):
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=batch_size,
        random_state=random_state,
        verbose=verbose,
        n_init='auto'
    )
    kmeans.fit(X)
    centroids = kmeans.cluster_centers_
    medoid_indices, _ = pairwise_distances_argmin_min(
        centroids, X, metric='euclidean'
    )
    medoids = X[medoid_indices]
    return medoid_indices

def active_learning(X, y, size, seed, model, batch):
    '''Use active learning to choose training data.'''

    # Randomly choose initial sample.
    idx = [i for i in range(X.shape[0])]
    np.random.seed(seed)
    np.random.shuffle(idx)
    batch_size = int(0.1 * size)
    chosen_idx = idx[0:batch_size]

    # Iteratively choose new data.
    for iter in range(9):
        print(f'Working on batch {iter + 1} / 10.')
        surrogate = get_model(model)
        surrogate.train(X[chosen_idx], y[chosen_idx], tune=True)
        y_std = surrogate.get_uncertainties(X)
        new_idx = select_batch(
            batch, batch_size,
            X, y, chosen_idx,
            surrogate, y_std
        )
        for idx in new_idx:
            chosen_idx.append(idx)
    
    return chosen_idx

if __name__ == '__main__':

    import argparse
    import time
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', default='jarvis18')
    parser.add_argument('--target', default='e_form')
    parser.add_argument('--size', type=int, default=100)
    parser.add_argument('--strategy', default='sf')
    parser.add_argument('--sampler', default='random')
    parser.add_argument('--model', default='gp')
    parser.add_argument('--batch', default='topk')
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    # Load dataset.
    start = time.perf_counter()
    np.random.seed(args.seed)
    dataset = np.load(f'./databases/{args.database}-{args.target}.npy', allow_pickle=True)
    X = dataset[:,0:-1]
    y = dataset[:,-1]

    # Scale inputs.
    X = StandardScaler().fit_transform(X)

    # Generate dataset with appropriate methods.
    if args.strategy == 'sf' and args.sampler == 'random':
        idx = [i for i in range(X.shape[0])]
        np.random.shuffle(idx)
        chosen_idx = idx[0:args.size]
    elif args.strategy == 'sf' and args.sampler == 'cluster':
        chosen_idx = large_scale_kmedoids(X, args.size, random_state=args.seed)
    else:
        chosen_idx = active_learning(X, y, args.size, args.seed, args.model, args.batch)

    # Save chosen measurements to file.
    chosen_idx = np.array(chosen_idx, dtype=np.int32)
    if args.strategy == 'sf':
        file_str = f'{args.database}-{args.target}-sf-{args.sampler}-{args.seed}'
    else:
        file_str = f'{args.database}-{args.target}-al-{args.model}-{args.batch}-{args.seed}'
    print(f'Size of dataset: {len(chosen_idx)}')
    print(chosen_idx)
    np.save(f'./datasets/size_{args.size}/{file_str}', chosen_idx)
    end = time.perf_counter()
    print(f'{end-start:.3f} seconds for dataset generation.')