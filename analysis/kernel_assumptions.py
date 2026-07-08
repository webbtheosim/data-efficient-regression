import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
import torch
import pickle

import sys
sys.path.append('..')
from models import get_model

import functools
print = functools.partial(print, flush=True)

def compute_metrics(path):
    print(f'Working on path: {path}')

    # Get dataset information.
    vals = path[:-4].split('-')
    task_name = vals[0]
    strat = vals[1]
    space = vals[2]
    if strat == 'sf':
        model_name = 'none'
        batch_strat = 'none'
        num = int(vals[3])
        seed = int(vals[4])
    else:
        model_name = vals[3]
        batch_strat = vals[4]
        num = int(vals[5])
        seed = int(vals[6])

    # Load the full dataset.
    task = pd.read_csv(f'../survey/tasks/{task_name}.csv', index_col=0)
    X = task.iloc[:,0:-1].to_numpy()
    y = task.iloc[:,-1].to_numpy()
    X = StandardScaler().fit_transform(X)
    chosen_idx = np.load(f'../survey/datasets/size_{num}/{path}')
    X_train = X[chosen_idx]
    y_train = y[chosen_idx]
    surrogate = pickle.load(open(f'./gp_models/{task_name}.pkl', 'rb'))

    # Compute contribution of each training point on each domain point.
    X_test = torch.tensor(X, dtype=torch.float64)
    X_train = torch.tensor(X_train, dtype=torch.float64)
    y_train = torch.tensor(y_train, dtype=torch.float64).reshape(-1,1)
    metrics = []
    with torch.no_grad():

        # Compute clustering criterion.
        d = cdist(X, X[chosen_idx], metric='sqeuclidean')
        neigh_dist = np.min(d, axis=1)
        wcss = np.sum(neigh_dist).item()
        metrics.append(wcss)

        # Compute validity of uncorrelated assumption.
        k_train = surrogate.model.covar_module(X_train, X_train).evaluate()
        noise = surrogate.model.likelihood.noise.item()
        diag = torch.sum(k_train.diag()).item()
        nondiag = k_train - k_train[0,0] * torch.eye(k_train.shape[0])
        nondiag = torch.sum(nondiag, dim=(0,1)).item()
        metrics.append(diag)
        metrics.append(nondiag)

        # Compute actual coverage metric.
        k_train = k_train + noise * torch.eye(k_train.shape[0])
        k_train_inv = torch.linalg.inv(k_train)
        k_test_left = surrogate.model.covar_module(X_test, X_train).evaluate()
        k_test_right = k_test_left.transpose(0,1)
        k_real = torch.matmul(k_train_inv, k_test_right)
        k_real = torch.matmul(k_test_left, k_real)
        k_real = k_real.diag()
        metric_real = torch.sum(k_real).item()
        metrics.append(metric_real)

        # Compute nearest neighbor coverage metric.
        k_train_near = k_train * torch.eye(k_train.shape[0], dtype=torch.float64)
        k_train_inv_near = torch.linalg.inv(k_train_near)
        max_idx = torch.argmax(k_test_left, dim=1)
        k_test_left_near = torch.zeros_like(k_test_left)
        k_test_left_near[torch.arange(k_test_left.shape[0]), max_idx] = k_test_left[torch.arange(k_test_left.shape[0]), max_idx]
        max_idx = torch.argmax(k_test_right, dim=0)
        k_test_right_near = k_test_left_near.transpose(0,1)
        k_near = torch.matmul(k_train_inv_near, k_test_right_near)
        k_near = torch.matmul(k_test_left_near, k_near)
        k_near = k_near.diag()
        metric_near = torch.sum(k_near).item()
        metrics.append(metric_near)

        # Compute perturbative coverage metric (include all terms above a cutoff...)
        k_diag = k_train * torch.eye(k_train.shape[0], dtype=torch.float64)
        k_off_diag = k_train - k_diag

        factor = 0.98
        k_cut = min(torch.max(k_off_diag) * factor, torch.max(k_diag))
        k_train_pert = torch.where(k_train >= k_cut, k_train, 0.0)
        k_train_inv_pert = torch.linalg.inv(k_train_pert)
        k_test_left_pert = torch.where(k_test_left >= k_cut, k_test_left, 0.0)
        k_test_right_pert = k_test_left_pert.transpose(0,1)
        k_pert = torch.matmul(k_train_inv_pert, k_test_right_pert)
        k_pert = torch.matmul(k_test_left_pert, k_pert)
        k_pert = k_pert.diag()
        metric_pert = torch.sum(k_pert).item()
        metrics.append(metric_pert)

        factor = 0.95
        k_cut = min(torch.max(k_off_diag) * factor, torch.max(k_diag))
        k_train_pert = torch.where(k_train >= k_cut, k_train, 0.0)
        k_train_inv_pert = torch.linalg.inv(k_train_pert)
        k_test_left_pert = torch.where(k_test_left >= k_cut, k_test_left, 0.0)
        k_test_right_pert = k_test_left_pert.transpose(0,1)
        k_pert = torch.matmul(k_train_inv_pert, k_test_right_pert)
        k_pert = torch.matmul(k_test_left_pert, k_pert)
        k_pert = k_pert.diag()
        metric_pert = torch.sum(k_pert).item()
        metrics.append(metric_pert)

        factor = 0.90
        k_cut = min(torch.max(k_off_diag) * factor, torch.max(k_diag))
        k_train_pert = torch.where(k_train >= k_cut, k_train, 0.0)
        k_train_inv_pert = torch.linalg.inv(k_train_pert)
        k_test_left_pert = torch.where(k_test_left >= k_cut, k_test_left, 0.0)
        k_test_right_pert = k_test_left_pert.transpose(0,1)
        k_pert = torch.matmul(k_train_inv_pert, k_test_right_pert)
        k_pert = torch.matmul(k_test_left_pert, k_pert)
        k_pert = k_pert.diag()
        metric_pert = torch.sum(k_pert).item()
        metrics.append(metric_pert)

    # Write to file.
    with open('data/metrics.csv', 'a') as handle:
        write_str = f'{task_name},{strat},{space},{model_name},{batch_strat},{num},{seed}'
        for m in metrics:
            write_str += f',{m}'
        write_str += '\n'
        handle.write(write_str)

if __name__ == '__main__':

    from joblib import Parallel, delayed
    import os
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=100)
    args = parser.parse_args()
    Parallel(n_jobs=5)(delayed(compute_metrics)(path) for path in os.listdir(f'../survey/datasets/size_{args.size}/'))