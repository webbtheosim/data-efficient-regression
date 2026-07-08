import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

import sys
sys.path.append('../../survey')
from models import get_model

import functools
print = functools.partial(print, flush=True)

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task')
    args = parser.parse_args()
    print(f'Training GP for task: {args.task}')

    task = pd.read_csv(f'../../survey/tasks/{args.task}.csv', index_col=0)
    X = task.iloc[:,0:-1].to_numpy()
    y = task.iloc[:,-1].to_numpy()
    X = StandardScaler().fit_transform(X)

    if X.shape[0] > 1000:
        idx = [i for i in range(X.shape[0])]
        np.random.shuffle(idx)
        train_idx = idx[0:1000]
        X = X[train_idx]
        y = y[train_idx]

    surrogate = get_model(model_name='gp_ard')
    surrogate.train(X, y, tune=True, print_progress=True)
    pickle.dump(surrogate, open(f'./{args.task}.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)