import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import time

from samplers import sample
from models import *
from batch_selection import *
from utils import mastml_tasks, mordred_tasks

import functools
print = functools.partial(print, flush=True)

def run_space_filling(sampler, X, size, seed):
    '''
        Method for choosing training dataset using the specified
        space-filling algorithm. Returns a np.int array of the chosen
        indices associated with the provided dataset (X, y).
    '''
    chosen_idx = sample(sampler, X, size, seed)
    return np.array(chosen_idx, dtype=np.int32)

def _relevant_dimensions(X,y):
    '''
        Helper method for determining irrelevant features for inputs X and 
        labels y. Specifically, we remove those features with feature importances
        more than an order of magnitude less than the most important feature, where
        feature importances are determined from a single random forest.
    '''
    model = RandomForestRegressor()
    model.fit(X,y)
    feat_import = model.feature_importances_
    cutoff = 0.01 * np.max(feat_import)
    chosen_feat = [i for i in range(X.shape[1]) if feat_import[i] > cutoff]
    return chosen_feat

def run_adaptive_space_filling(X, y, size, seed):
    '''
        Method for chosing a training dataset using sequential k-medoids
        sampling applied to adaptively determined subspaces. Returns a np.int
        array of the chosen indicates associated with the provided (X, y).
    '''

    # Choose an initial sample to initiative data selection.
    batch_size = int(0.1 * size)
    print(f'Choosing points for round 1 / 10.')
    chosen_idx = sample('medoids', X, batch_size, seed)

    # Start data selection loop.
    for round in range(9):
        print(f'Choosing points for round {round + 2} / 10.')

        # Determine relevant features.
        X_sample = X[chosen_idx]
        y_sample = y[chosen_idx]
        chosen_feat = _relevant_dimensions(X_sample, y_sample)
        print(f'{len(chosen_feat)} features chosen...')

        # Choose points from the chosen subspace.
        new_idx = sample(
            name='fixed-medoids',
            domain=X[:,chosen_feat],
            size=batch_size,
            seed=seed,
            fixed_idx=chosen_idx
        )

        # Add new points to the selection.
        for idx in new_idx:
            chosen_idx.append(idx)

    # Compute final set of relevant dimensions prior to model training.
    X_sample = X[chosen_idx]
    y_sample = y[chosen_idx]
    chosen_feat = _relevant_dimensions(X_sample, y_sample)

    return np.array(chosen_idx, dtype=np.int32), chosen_feat

def run_adaptive_active_learning(sampler, model, X, y, batch_strat, size, seed):
    '''
        Method for choosing training dataset using the specified
        active learning and batch selection algorithm. Returns a 
        np.int array of the chosen indices associated with the 
        provided dataset (X, y). This specific form of active learning makes
        use of adaptive feature selection based on a random forest.
    '''

    # Choose an initial sample to initiate active learning.
    batch_size = int(0.1 * size)
    print(f'Choosing points for round 1 / 10.')
    chosen_idx = sample(sampler, X, batch_size, seed)

    # Define model for guiding active learning.
    surrogate = get_model(model)

    # Start active learning loop.
    for round in range(9):
        print(f'Choosing points for round {round + 2} / 10.')

        # Determine relevant features.
        X_train = X[chosen_idx]
        y_train = y[chosen_idx]
        chosen_feat = _relevant_dimensions(X_train, y_train)
        print(f'Number of chosen features: {len(chosen_feat)}')

        # Compute uncertainties in model over design space.
        surrogate.train(X_train[:,chosen_feat], y_train, tune=True)
        y_std = surrogate.get_uncertainties(X[:,chosen_feat])

        # Choose new points.
        new_idx = select_batch(
            batch_strat, batch_size,
            X, y, chosen_idx,
            surrogate, y_std, chosen_feat
        )

        # Add new points to the selection.
        for idx in new_idx:
            chosen_idx.append(idx)

    # Compute final set of relevant dimensions prior to model training.
    X_train = X[chosen_idx]
    y_train = y[chosen_idx]
    chosen_feat = _relevant_dimensions(X_train, y_train)

    return np.array(chosen_idx, dtype=np.int32), chosen_feat
    
def run_active_learning(sampler, model, X, y, batch_strat, size, seed):
    '''
        Method for choosing training dataset using the specified
        active learning and batch selection algorithm. Returns a 
        np.int array of the chosen indices associated with the 
        provided dataset (X, y).
    '''

    # Choose an initial sample to initiate active learning.
    batch_size = int(0.1 * size)
    print(f'Choosing points for round 1 / 10.')
    chosen_idx = sample(sampler, X, batch_size, seed)

    # Define model for guiding active learning.
    surrogate = get_model(model)

    # Start active learning loop.
    for round in range(9):
        print(f'Choosing points for round {round + 2} / 10.')

        # Compute uncertainties in model over design space.
        X_train = X[chosen_idx]
        y_train = y[chosen_idx]
        surrogate.train(X_train, y_train, tune=True)
        y_std = surrogate.get_uncertainties(X)

        # Choose new points.
        new_idx = select_batch(
            batch_strat, batch_size,
            X, y, chosen_idx,
            surrogate, y_std
        )

        # TESTING: Visualize model predictions and uncertainties.
        # y_pred = surrogate.predict(X)
        # visualize_results(X, y, chosen_idx, y_pred, y_std, new_idx)

        # Add new points to the selection.
        for idx in new_idx:
            chosen_idx.append(idx)

    return np.array(chosen_idx, dtype=np.int32)

def visualize_results(X, y, chosen_idx, y_pred, y_std, new_idx):
    '''
        Method for visualizing the current status of active learning.
    '''

    plt.rcParams['font.size'] = 12
    plt.rcParams['font.family'] = 'Helvetica Neue'
    plt.rcParams['axes.linewidth'] = 1.5
    fig, axs = plt.subplots(1,4,figsize=(12,3))

    vmin = np.min(np.hstack((y, y_pred)))
    vmax = np.max(np.hstack((y, y_pred)))

    # Plot currently chosen data on ground-truth plot.
    axs[0].scatter(X[:,0], X[:,1], s=10.0, c=y, cmap=plt.get_cmap('Blues_r'), zorder=1, vmin=vmin, vmax=vmax)
    axs[0].scatter(X[chosen_idx, 0], X[chosen_idx, 1], s=30.0, color='tomato', edgecolors='black', linewidth=0.8, zorder=2)

    # Plot current predictions of data.
    axs[1].scatter(X[:,0], X[:,1], s=10.0, c=y_pred, cmap=plt.get_cmap('Blues_r'), zorder=1, vmin=vmin, vmax=vmax)
    axs[1].scatter(X[chosen_idx, 0], X[chosen_idx, 1], s=30.0, color='tomato', edgecolors='black', linewidth=0.8, zorder=2)

    # Plot current uncertainties of data.
    axs[2].scatter(X[:,0], X[:,1], s=10.0, c=y_std, cmap=plt.get_cmap('plasma'), zorder=1)
    axs[2].scatter(X[chosen_idx, 0], X[chosen_idx, 1], s=30.0, color='white', edgecolors='black', linewidth=0.8, zorder=2)

    # Plot newly chosen points.
    axs[3].scatter(X[:,0], X[:,1], s=10.0, c=y_std, cmap=plt.get_cmap('plasma'), zorder=1)
    axs[3].scatter(X[chosen_idx, 0], X[chosen_idx, 1], s=30.0, color='white', edgecolors='black', linewidth=0.8, zorder=2)
    axs[3].scatter(X[new_idx, 0], X[new_idx, 1], s=30.0, color='tomato', edgecolors='black', linewidth=0.8, zorder=2)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    # Gather user input.
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', choices=['sf', 'al', 'sf_adaptive', 'al_adaptive'], default='sf')
    parser.add_argument(
        '--sampler', 
        choices=[
            'random', 'maximin', 'medoids', 
            'max_entropy', 'vendi'
        ], 
        default='random'
    )
    parser.add_argument(
        '--model',
        choices=[
            'nn', 'knn', 'gp', 'gp_ard',
            'rf', 'xgb', 'sv'
        ],
        default='nn'
    )
    parser.add_argument('--task', default='muller_brown')
    parser.add_argument('--size', type=int, default=50)
    parser.add_argument(
        '--batch_strat', 
        choices=[
            'topk',
            'hallucinate',
            'pareto',
            'cluster_margin'
        ], 
        default='topk'
    )
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--full', action='store_true')
    args = parser.parse_args()

    # Prepare dataset.
    if not args.full:
        task = pd.read_csv(f'./tasks/{args.task}.csv', index_col=0)
        X = task.iloc[:,0:-1].to_numpy()
        y = task.iloc[:,-1].to_numpy()
        X = StandardScaler().fit_transform(X) # Scale because we know this information beforehand.
    else:
        if args.task in mastml_tasks:
            task_path = f'mastml/{args.task}.csv'
        elif args.task in mordred_tasks:
            task_path = f'mordred/{args.task}.csv'
        else:
            task_path = f'{args.task}.csv'
        task = pd.read_csv(f'./tasks/{task_path}', index_col=0)
        X = task.iloc[:,0:-1].to_numpy()
        y = task.iloc[:,-1].to_numpy()
        X = StandardScaler().fit_transform(X) # Scale because we know this information beforehand.
    print(f'Dataset dimensionality: {X.shape[1]}')

    # Execute space-filling, if requested.
    if args.strategy == 'sf':
        start = time.perf_counter()
        chosen_idx = run_space_filling(args.sampler, X, args.size, args.seed)
        end = time.perf_counter()
        print(f'{end-start:3f} seconds for {args.sampler} of {args.size} points.')

        # Save chosen indices to file.
        file_str = f'{args.task}-sf-{args.sampler}-{args.size}-{args.seed}.npy'
        np.save(f'./survey-datasets/{file_str}', chosen_idx)

    elif args.strategy == 'sf_adaptive':
        start = time.perf_counter()
        chosen_idx, chosen_feat = run_adaptive_space_filling(X, y, args.size, args.seed)
        end = time.perf_counter()
        print(f'{end-start:3f} seconds for sf_adaptive of {args.size} points.')

        # Save chosen indices to file.
        file_str = f'{args.task}-sf_adaptive-{args.size}-{args.seed}.npy'
        np.save(f'./survey-datasets/{file_str}', chosen_idx)
        np.save(f'./chosen_features/{file_str}', chosen_feat)

    elif args.strategy == 'al_adaptive':
        start = time.perf_counter()
        chosen_idx, chosen_feat = run_adaptive_active_learning(args.sampler, args.model, X, y, args.batch_strat, args.size, args.seed)
        end = time.perf_counter()
        print(f'{end-start:.3f} seconds for {args.sampler}-{args.model}-{args.batch_strat} of {args.size} points.')

        # Save chosen indices to file.
        file_str = f'{args.task}-al_adaptive-{args.sampler}-{args.model}-{args.batch_strat}-{args.size}-{args.seed}.npy'
        np.save(f'./survey-datasets/{file_str}', chosen_idx)
        np.save(f'./chosen_features/{file_str}', chosen_feat)

    # Execute active learning, if requested.
    else:
        start = time.perf_counter()
        chosen_idx = run_active_learning(args.sampler, args.model, X, y, args.batch_strat, args.size, args.seed)
        end = time.perf_counter()
        print(f'{end-start:.3f} seconds for {args.sampler}-{args.model}-{args.batch_strat} of {args.size} points.')

        # Save chosen indices to file.
        file_str = f'{args.task}-al-{args.sampler}-{args.model}-{args.batch_strat}-{args.size}-{args.seed}.npy'
        np.save(f'./survey-datasets/{file_str}', chosen_idx)
