'''
    Objective: Compute coverage scores for high-dimensional datasets.
'''

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from utils import all_tasks, mordred_tasks, mastml_tasks

import functools
print = functools.partial(print, flush=True)

def compute_coverage(X, train_idx_path, size):
    '''
        Compute coverage of the given training set on the
        provided domain for the lengthscale l.
    '''

    # Load training set.
    print(f'Working on training set: {train_idx_path}...')
    train_idx = np.load(f'./datasets_raw/size_{args.size}_full/{train_idx_path}')

    # Compute WCSS.
    d = cdist(X, X[train_idx], metric='sqeuclidean')
    neigh_dist = np.min(d, axis=1)
    coverage = np.sum(neigh_dist)

    # Compute dimension-by-dimension distances.
    d = np.abs(X[:,None,:] - X[train_idx][None,:,:])
    max_val = np.max(d)
    d = np.where(d == 0, max_val, d)
    min_d = np.min(d, axis=0)
    d = np.sum(min_d, axis=0)
    avg_d = np.mean(d)

    # Save coverage to appropriate file.
    with open(f'metrics_{size}_high_d.csv', 'a') as handle:
        handle.write(f'{train_idx_path},{coverage},{avg_d}\n')

def _relevant_dimensions(X,y):
    '''
        Helper method for determining irrelevant features for inputs X and 
        labels y. Specifically, we remove those features with feature importances
        more than an order of magnitude less than the most important feature, where
        feature importances are determined from a single random forest.
    '''
    model = RandomForestRegressor(n_jobs=-1)
    model.fit(X,y)
    feat_import = model.feature_importances_
    cutoff = 0.01 * np.max(feat_import)
    chosen_feat = [i for i in range(X.shape[1]) if feat_import[i] > cutoff]
    return chosen_feat

if __name__ == '__main__':

    import argparse
    import os
    import pandas as pd
    import pickle
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='freesolv')
    parser.add_argument('--size', type=int, default=100)
    args = parser.parse_args()

    print(f'Loading task...')

    if args.task in mordred_tasks:
        task = pd.read_csv(f'./tasks/mordred/{args.task}.csv', index_col=0)
        X = task.iloc[:,0:-1].to_numpy()
        y = task.iloc[:,-1].to_numpy()
        X = StandardScaler().fit_transform(X)
        print(f'Task size: {X.shape}')
    elif args.task in mastml_tasks:
        task = pd.read_csv(f'./tasks/mastml/{args.task}.csv', index_col=0)
        X = task.iloc[:,0:-1].to_numpy()
        y = task.iloc[:,-1].to_numpy()
        X = StandardScaler().fit_transform(X)
        print(f'Task size: {X.shape}')
    else:
        task = pd.read_csv(f'./tasks/{args.task}.csv', index_col=0)
        X = task.iloc[:,0:-1].to_numpy()
        y = task.iloc[:,-1].to_numpy()
        X = StandardScaler().fit_transform(X)
        print(f'Task size: {X.shape}')

    # relevant_features = _relevant_dimensions(X,y)
    # X = X[:,relevant_features]

    from joblib import Parallel, delayed
    print('Computing coverages...')
    Parallel(n_jobs=96)(
        delayed(compute_coverage)(X, train_idx_path, args.size)
        for train_idx_path in os.listdir(f'../survey-datasets/size_100_high_d/') if args.task in train_idx_path
    )
    print('Finished.')