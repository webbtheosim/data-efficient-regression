import numpy as np
import os
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

import sys
sys.path.append('')
from models import GP

import functools
print = functools.partial(print, flush=True)

def compute_lengthscale(X, y):
    '''
        Compute lengthscale optimized for GP fitting
        with an RBF kernel.
    '''
    if X.shape[0] > 1000:
        idx = [i for i in range(X.shape[0])]
        np.random.seed(1)
        np.random.shuffle(idx)
        X = X[idx[0:1000]]
        y = y[idx[0:1000]]
    gp = GP(isotropic=True)
    gp.train(X, y, tune=True, print_progress=True)
    return gp.model.covar_module.base_kernel.lengthscale.item()

def compute_coverage(X, train_idx_path, l, size):
    '''
        Compute coverage of the given training set on the
        provided domain for the lengthscale l.
    '''

    # Load training set.
    print(f'Working on training set: {train_idx_path}...')
    train_idx = np.load(f'../survey-datasets/size_{size}/{train_idx_path}')

    # Compute WCSS.
    d = cdist(X, X[train_idx], metric='sqeuclidean')
    neigh_dist = np.min(d, axis=1)
    wcss = np.sum(neigh_dist)

    # Compute coverage.
    k = np.exp(-0.5 * neigh_dist / (l ** 2))
    k2 = k * k
    coverage = np.sum(k2, axis=0)

    # Compute full pairwise information.
    pk = np.exp(-0.5 * d / (l ** 2))
    pairwise_kernel = pk * pk
    pairwise_metric = np.sum(pairwise_kernel, axis=(0,1))

    # Compute entropy (per Schwalbe-Koda et al.)
    ent = np.mean(pk, axis=1)
    ent = np.log(ent + 1e-30)
    entropy = -np.mean(ent)

    # Save coverage to appropriate file.
    with open(f'coverages_{size}.csv', 'a') as handle:
        handle.write(f'{train_idx_path},{wcss},{coverage},{pairwise_metric},{entropy}\n')

if __name__ == '__main__':

    import argparse
    import pickle
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='muller_brown')
    parser.add_argument('--size', type=int, default=200)
    args = parser.parse_args()

    print(f'Loading task...')
    task = pd.read_csv(f'./tasks/{args.task}.csv', index_col=0)
    X = task.iloc[:,0:-1].to_numpy()
    y = task.iloc[:,-1].to_numpy()
    X = StandardScaler().fit_transform(X)
    print(f'Task size: {X.shape}')

    l = None
    if not os.path.exists('./task_lengthscales.pkl'):
        lengthscale_database = {}
        pickle.dump(lengthscale_database, open('task_lengthscales.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    lengthscale_database = pickle.load(open('task_lengthscales.pkl', 'rb'))
    if args.task in lengthscale_database.keys():
        l = lengthscale_database[args.task]
    if l is None:
        print('Computing lengthscale...')
        l = compute_lengthscale(X, y)
        print(f'Computed lengthscale: {l}')
        lengthscale_database[args.task] = l
        pickle.dump(lengthscale_database, open('task_lengthscales.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    from joblib import Parallel, delayed
    print('Computing coverages...')
    Parallel(n_jobs=96)(
        delayed(compute_coverage)(X, train_idx_path, l, args.size)
        for train_idx_path in os.listdir(f'../survey-datasets/size_{args.size}/') if args.task in train_idx_path
    )
    print('Finished.')
