import numpy as np
import pandas as pd
import rdkit
import rdkit.Chem as Chem
from scipy.spatial.distance import cdist
from sklearn.metrics import (
    r2_score, 
    mean_absolute_error, 
    mean_absolute_percentage_error, 
    root_mean_squared_error
)
from scipy.stats import spearmanr, pearsonr
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
import sys
sys.path.append('..')
from samplers import sample

import functools
print = functools.partial(print, flush=True)

def run_active_learning(graphs, method, size, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Set up active learning.
    batch_size = int(0.1 * size)
    surrogate = build_gnn(f'gnn-{method}', in_feats=graphs[0].x.shape[1])
    test_loader = pyg_DataLoader(graphs, batch_size=64, shuffle=False)

    # Choose an initial sample to initiate active learning.
    print(f'Choosing points for round 1 / 10.')
    graph_idx = [i for i in range(len(graphs))]
    np.random.shuffle(graph_idx)
    chosen_idx = graph_idx[0:batch_size]

    # Start active learning loop.
    for round in range(9):
        print(f'Choosing points for round {round + 2} / 10.')

        # Build training, validation, and testing dataloaders.
        train_idx = chosen_idx[0:int(0.9 * len(chosen_idx))]
        val_idx = chosen_idx[int(0.9 * len(chosen_idx)):]
        train_data = [graphs[i] for i in train_idx]
        val_data = [graphs[i] for i in val_idx]
        train_loader = pyg_DataLoader(train_data, batch_size=64, shuffle=True)
        val_loader = pyg_DataLoader(val_data, batch_size=64, shuffle=False)
    
        # Train surrogate model and select new batch.
        surrogate.train(train_loader, val_loader, tune=False)
        temp_idx = [i for i in chosen_idx]
        new_idx = select_batch(method, batch_size, test_loader, temp_idx, surrogate)
        for idx in new_idx:
            chosen_idx.append(idx)

    return np.array(chosen_idx, dtype=np.int32)

def run_hallucinate_al(graphs, method, size, seed):
    '''
        A version of active learning that implements hallucination for
        batch selection instead of the embedding-based techniques.
    '''
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Set up active learning.
    batch_size = int(0.1 * size)
    surrogate = build_gnn(f'gnn-{method}', in_feats=graphs[0].x.shape[1])
    test_loader = pyg_DataLoader(graphs, batch_size=64, shuffle=False)

    # Choose an initial sample to initiate active learning.
    print(f'Choosing points for round 1 / 10.')
    graph_idx = [i for i in range(len(graphs))]
    np.random.shuffle(graph_idx)
    chosen_idx = graph_idx[0:batch_size]

    # Start active learning loop.
    for round in range(9):
        print(f'Choosing points for round {round + 2} / 10.')

        # Build training, validation, and testing dataloaders.
        np.random.shuffle(chosen_idx)
        train_idx = chosen_idx[0:int(0.9 * len(chosen_idx))]
        val_idx = chosen_idx[int(0.9 * len(chosen_idx)):]
        train_data = [graphs[i] for i in train_idx]
        val_data = [graphs[i] for i in val_idx]
        train_loader = pyg_DataLoader(train_data, batch_size=64, shuffle=True)
        val_loader = pyg_DataLoader(val_data, batch_size=64, shuffle=False)
    
        # Train surrogate model and select new batch.
        new_idx = []
        for batch_num in range(batch_size):
            print(f'Choosing batch point: {batch_num + 1}.')
            surrogate.train(train_loader, val_loader, tune=False)
            y_std = surrogate.get_uncertainties(test_loader)
            idx = np.argmax(y_std).item()
            new_idx.append(idx)
            y_pred = surrogate.predict_single_graph(graphs[idx])
            new_graph = Data(
                x=graphs[idx].x,
                edge_index=graphs[idx].edge_index,
                smiles=graphs[idx].smiles,
                y=torch.tensor([y_pred])
            )
            train_data.append(new_graph)
            train_loader = pyg_DataLoader(train_data, batch_size=64, shuffle=True)
            val_loader = pyg_DataLoader(val_data, batch_size=64, shuffle=False)

        # Add new points to the training set.
        for i in new_idx:
            chosen_idx.append(i)

    return np.array(chosen_idx, dtype=np.int32)

def run_pareto_al(graphs, method, size, seed):
    '''
        A version of active learning that implements Pareto for
        batch selection instead of the embedding-based techniques.
    '''
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Set up active learning.
    batch_size = int(0.1 * size)
    surrogate = build_gnn(f'gnn-{method}', in_feats=graphs[0].x.shape[1])
    test_loader = pyg_DataLoader(graphs, batch_size=64, shuffle=False)

    # Choose an initial sample to initiate active learning.
    print(f'Choosing points for round 1 / 10.')
    graph_idx = [i for i in range(len(graphs))]
    np.random.shuffle(graph_idx)
    chosen_idx = graph_idx[0:batch_size]

    # Start active learning loop.
    for round in range(9):
        print(f'Choosing points for round {round + 2} / 10.')

        # Build training, validation, and testing dataloaders.
        np.random.shuffle(chosen_idx)
        train_idx = chosen_idx[0:int(0.9 * len(chosen_idx))]
        val_idx = chosen_idx[int(0.9 * len(chosen_idx)):]
        train_data = [graphs[i] for i in train_idx]
        val_data = [graphs[i] for i in val_idx]
        train_loader = pyg_DataLoader(train_data, batch_size=64, shuffle=True)
        val_loader = pyg_DataLoader(val_data, batch_size=64, shuffle=False)
    
        # Train surrogate model and select new batch.
        new_idx = []
        weights = np.linspace(0, 1, num=batch_size)
        for batch_num in range(batch_size):
            print(f'Choosing batch point: {batch_num + 1}.')
            surrogate.train(train_loader, val_loader, tune=False)

            # Get uncertainty of surrogate model.
            y_std = surrogate.get_uncertainties(test_loader)

            # Get distances in embedding space.
            embeddings_all = surrogate.get_embeddings(test_loader)
            embeddings_train = surrogate.get_embeddings(train_loader)
            d = cdist(embeddings_all, embeddings_train)
            d = np.min(d, axis=1)
            d_sc = d / np.max(d)

            # Compute acquistion function.
            w = weights[batch_num]
            y_acq = (1 - w) * y_std + w * d_sc

            # Add best point to the new batch.
            idx = np.argmax(y_acq).item()
            new_idx.append(idx)
            y_pred = surrogate.predict_single_graph(graphs[idx])
            new_graph = Data(
                x=graphs[idx].x,
                edge_index=graphs[idx].edge_index,
                smiles=graphs[idx].smiles,
                y=torch.tensor([y_pred])
            )
            train_data.append(new_graph)
            train_loader = pyg_DataLoader(train_data, batch_size=64, shuffle=True)
            val_loader = pyg_DataLoader(val_data, batch_size=64, shuffle=False)

        # Add new points to the training set.
        for i in new_idx:
            chosen_idx.append(i)

    return np.array(chosen_idx, dtype=np.int32)

def select_batch(method, batch_size, test_loader, chosen_idx, surrogate):
    '''Wrapper for various batch selection methods.'''
    if method == 'ensemble':
        X = surrogate.get_embeddings(test_loader)
        k_m = int(0.05 * X.shape[0]) # 5% of the domain is considered.
        y_std = surrogate.get_uncertainties(test_loader)
        y_std[chosen_idx] = -np.inf
        candidate_idx = np.argsort(-y_std).reshape(-1)[0:k_m]
        X_candidate = X[candidate_idx]
        subsample_idx = sample('medoids', X_candidate, batch_size, seed=1)
        new_idx = [candidate_idx[idx] for idx in subsample_idx]
        return new_idx
    elif method == 'top':
        y_std = surrogate.get_uncertainties(test_loader)
        y_std[chosen_idx] = -np.inf
        new_idx = np.argwhere(y_std).reshape(-1)[-batch_size:]
        return new_idx.tolist()
    elif method == 'embedding' or method == 'penultimate':
        embeddings = surrogate.get_embeddings(test_loader)
        new_idx = sample(name='fixed-medoids', domain=embeddings,
            size=batch_size, seed=1, fixed_idx=chosen_idx)
        return new_idx
    else:
        raise Exception('This is not a viable batch selection method!')
    
def scale_data(train_data, val_data):
    '''Scaling labels in train and validation loaders.'''
    y_train = np.array([data.y.clone().detach().numpy() for data in train_data], dtype=np.float32)
    y_val = np.array([data.y.clone().detach().numpy() for data in val_data], dtype=np.float32)
    label_scaler = StandardScaler().fit(y_train.reshape(-1,1))
    y_train_sc = label_scaler.transform(y_train.reshape(-1,1)).reshape(-1)
    y_val_sc = label_scaler.transform(y_val.reshape(-1,1)).reshape(-1)
    for data_idx, data in enumerate(train_data):
        train_data[data_idx].y = torch.tensor([y_train_sc[data_idx]], dtype=torch.float32)
    for data_idx, data in enumerate(val_data):
        val_data[data_idx].y = torch.tensor([y_val_sc[data_idx]], dtype=torch.float32)
    return train_data, val_data, label_scaler

def read_smiles_dataset(task):
    '''
        Method for loading .csv files with different formats because I'm too lazy
        to manually change the appropriate files.
    '''
    if task in ['logp', 'polygas_CH4', 'polygas_CO2', 'polygas_H2', 'polygas_He', 'polygas_N2', 'polygas_O2']:
        df = pd.read_csv(f'../tasks/smiles/{task}.csv')
    else:
        df = pd.read_csv(f'../tasks/smiles/{task}.csv', index_col=0)
    return df

def compute_atomic_properties(smiles):
    '''
        Compute unique atomic properties that will be used for one-hot encoding
        the appropriate node representations for this list of SMILES.
    '''
    atom_types = []
    degrees = []
    total_degrees = []
    explicit_valences = []
    implicit_valences = []
    total_valences = []
    implicit_Hs = []
    total_Hs = []
    formal_charges = []
    hybridizations = []
    for smi_idx, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi, sanitize=True)
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        for atom in mol.GetAtoms():
            val = atom.GetSymbol()
            if val not in atom_types:
                atom_types.append(val)
            val = atom.GetDegree()
            if val not in degrees:
                degrees.append(val)
            val = atom.GetTotalDegree()
            if val not in total_degrees:
                total_degrees.append(val)
            val = atom.GetValence(Chem.rdchem.ValenceType.EXPLICIT)
            if val not in explicit_valences:
                explicit_valences.append(val)
            val = atom.GetValence(Chem.rdchem.ValenceType.IMPLICIT)
            if val not in implicit_valences:
                implicit_valences.append(val)
            val = atom.GetTotalValence()
            if val not in total_valences:
                total_valences.append(val)
            val = atom.GetNumImplicitHs()
            if val not in implicit_Hs:
                implicit_Hs.append(val)
            val = atom.GetTotalNumHs()
            if val not in total_Hs:
                total_Hs.append(val)
            val = atom.GetFormalCharge()
            if val not in formal_charges:
                formal_charges.append(val)
            val = atom.GetHybridization().name
            if val not in hybridizations:
                hybridizations.append(val)
    return (atom_types, degrees, total_degrees, explicit_valences, implicit_valences,
            total_valences, implicit_Hs, total_Hs, formal_charges, hybridizations)

def evaluate_gnn(graphs, chosen_idx):
    '''Evaluating GNN on chosen dataset.'''
    train_data = [graphs[i] for i in chosen_idx[0:int(0.9 * len(chosen_idx))]]
    val_data = [graphs[i] for i in chosen_idx[int(0.9 * len(chosen_idx)):]]
    # train_data_sc, val_data_sc, label_scaler = scale_data(train_data, val_data)
    # train_loader = pyg_DataLoader(train_data_sc, batch_size=64, shuffle=True)
    # val_loader = pyg_DataLoader(val_data_sc, batch_size=64, shuffle=False)
    train_loader = pyg_DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = pyg_DataLoader(val_data, batch_size=64, shuffle=False)
    test_loader = pyg_DataLoader(graphs, batch_size=64, shuffle=False)
    model = build_gnn(f'gnn-embedding', in_feats=train_data[0].x.shape[1])
    model.train(train_loader, val_loader)
    y_pred = model.predict(test_loader).reshape(-1)
    y_test = np.array([data.y.clone().detach().numpy() for data in graphs], dtype=np.float32).reshape(-1)
    # y_test = label_scaler.transform(y_test.reshape(-1,1)).reshape(-1)
    result = [0, 0, 0, 0, 0, 0]
    result[0] = r2_score(y_test, y_pred)
    result[1] = mean_absolute_error(y_test, y_pred)
    result[2] = root_mean_squared_error(y_test, y_pred)
    result[3] = (root_mean_squared_error(y_test, y_pred) / np.std(y_test))
    result[4] = pearsonr(y_test, y_pred).statistic
    result[5] = spearmanr(y_test, y_pred).statistic
    return result

if __name__ == '__main__':

    import argparse
    import os
    import pandas as pd
    import time

    # Get user input for algorithm specification.
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['ensemble', 'embedding', 'penultimate', 'random', 'hallucinate', 'top', 'pareto'], default='ensemble')
    parser.add_argument('--task', default='logp')
    parser.add_argument('--size', type=int, default=100)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    # Develop viable graph representation for this dataset.
    df = read_smiles_dataset(args.task)
    smiles = df['smiles'].tolist()
    labels = df.iloc[:,-1].to_numpy()
    labels = StandardScaler().fit_transform(labels.reshape(-1,1)).reshape(-1)

    # Choose a quareter of the dataset size, capped at 1000 and floored at 100.
    size = 100
    print(f'Size for task {args.task}: {size}')

    # Convert all SMILES datasets to appropriate format.
    print('Converting SMILES strings to graphs...')
    start = time.perf_counter()
    atomic_property_values = compute_atomic_properties(smiles)
    graphs = [molecular_graph_featurizer(smi, label, atomic_property_values) for smi, label in zip(smiles, labels)]
    end = time.perf_counter()
    print(f'{end-start:.3f} seconds for converting SMILES to {len(graphs)} graphs.')

    # Run active learning.
    if args.method in ['ensemble', 'embedding', 'penultimate', 'top']:
        start = time.perf_counter()
        chosen_idx = run_active_learning(graphs.copy(), args.method, size, args.seed)
        end = time.perf_counter()
        print(f'{end-start:3f} seconds for graph AL of {size} points.')
    elif args.method in ['hallucinate']:
        start = time.perf_counter()
        chosen_idx = run_hallucinate_al(graphs.copy(), args.method, size, args.seed)
        end = time.perf_counter()
        print(f'{end-start:3f} seconds for graph AL of {size} points.')
    elif args.method in ['pareto']:
        start = time.perf_counter()
        chosen_idx = run_pareto_al(graphs.copy(), args.method, size, args.seed)
        end = time.perf_counter()
        print(f'{end-start:3f} seconds for graph AL of {size} points.')
    else:
        graph_idx = [i for i in range(len(graphs))]
        np.random.seed(args.seed)
        np.random.shuffle(graph_idx)
        chosen_idx = graph_idx[0:size]

    # Save generated dataset.
    file_str = f'{args.task}-{args.method}-{args.size}-{args.seed}.npy'
    np.save(f'./datasets/{file_str}', chosen_idx)

    # Evaluate GCN model on this dataset.
    start = time.perf_counter()
    result = evaluate_gnn(graphs, chosen_idx)
    end = time.perf_counter()
    print(f'{end-start:.3f} seconds for GCN evaluation.')

    print(f'{result[-1]:.5f}')

    # Write results to .csv file.
    with open('results.csv', 'a') as handle:
        write_str = f'{args.task},{args.method},none,none,none,{args.seed},{args.size}'
        for score in result:
            write_str = write_str + f',{score:.5f}'
        write_str = write_str + '\n'
        handle.write(write_str)