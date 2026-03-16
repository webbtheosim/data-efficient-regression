'''
    Objective: Compute coverage scores for high-dimensional datasets.
'''

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as pyg_DataLoader
from torch_geometric.nn import GCNConv, global_add_pool, BatchNorm

from gnn import build_gnn
from utils import (
    molecular_graph_featurizer,
    atom_featurizer,
    atom_props,
    match_patterns,
    structural_smarts,
    functional_group_smarts
)

from gen_data import (
    read_smiles_dataset,
    compute_atomic_properties,
    evaluate_gnn
)

import functools
print = functools.partial(print, flush=True)

def compute_coverage(graphs, train_idx_path, size):
    '''
        Compute coverage of the given training set on the
        provided domain for the lengthscale l.
    '''

    # Load training set.
    print(f'Working on training set: {train_idx_path}...')
    chosen_idx = np.load(f'./datasets_raw/{train_idx_path}')

    # Train GNN model on this training set.
    train_data = [graphs[i] for i in chosen_idx[0:int(0.9 * len(chosen_idx))]]
    val_data = [graphs[i] for i in chosen_idx[int(0.9 * len(chosen_idx)):]]
    train_loader = pyg_DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = pyg_DataLoader(val_data, batch_size=64, shuffle=False)
    test_loader = pyg_DataLoader(graphs, batch_size=64, shuffle=False)
    surrogate = build_gnn(f'gnn-embedding', in_feats=train_data[0].x.shape[1])
    surrogate.train(train_loader, val_loader)

    # Get embeddings for domain and training data.
    X = surrogate.get_embeddings(test_loader)
    graphs_train = [graphs[i] for i in chosen_idx]
    temp_loader = pyg_DataLoader(graphs_train, batch_size=64, shuffle=False)
    X_train = surrogate.get_embeddings(temp_loader)

    # Compute WCSS on embeddings.
    d = cdist(X, X_train, metric='sqeuclidean')
    neigh_dist = np.min(d, axis=1)
    coverage = np.sum(neigh_dist)

    # Compute coverage and save to file.
    with open(f'metric.csv', 'a') as handle:
        handle.write(f'{train_idx_path},{coverage}\n')

if __name__ == '__main__':

    import argparse
    import os
    import pickle
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='freesolv')
    parser.add_argument('--size', type=int, default=100)
    args = parser.parse_args()

    # Read in task SMILES and labels.
    print(f'Loading task...')
    df = read_smiles_dataset(args.task)
    smiles = df['smiles'].tolist()
    labels = df.iloc[:,-1].to_numpy()
    labels = StandardScaler().fit_transform(labels.reshape(-1,1)).reshape(-1)

     # Convert all SMILES datasets to appropriate format.
    print('Converting SMILES strings to graphs...')
    start = time.perf_counter()
    atomic_property_values = compute_atomic_properties(smiles)
    graphs = [molecular_graph_featurizer(smi, label, atomic_property_values) for smi, label in zip(smiles, labels)]
    end = time.perf_counter()
    print(f'{end-start:.3f} seconds for converting SMILES to {len(graphs)} graphs.')

    # Compute coverage scores in serial.
    for train_idx_path in os.listdir(f'./datasets/'):
        if args.task in train_idx_path:
            compute_coverage(graphs, train_idx_path, args.size)

    # # Compute coverage scores in parallel.
    # from joblib import Parallel, delayed
    # print('Computing coverages...')
    # Parallel(n_jobs=1)(
    #     delayed(compute_coverage)(graphs, train_idx_path, args.size)
    #     for train_idx_path in os.listdir(f'./datasets_raw/') if args.task in train_idx_path
    # )
    # print('Finished.')