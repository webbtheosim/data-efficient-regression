import gpytorch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error
)
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import torch
import xgboost as xgb

import functools
print = functools.partial(print, flush=True)

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

class RandomForest:

    def __init__(self):
        self.name = 'rf'

    def train(self, X, y):
        params = {
            'n_estimators': [100, 200, 300],
            'max_features': [0.5, 0.85, 1.0],
            'max_depth': [10, None]
        }
        rf = GridSearchCV(
            estimator=RandomForestRegressor(),
            param_grid=params,
            scoring='r2',
            cv=5,
            #verbose=5,
            n_jobs=1
        )
        rf.fit(X, y)
        self.model = rf.best_estimator_

    def predict(self, X):
        y = self.model.predict(X)
        return y

class GradientBoostedTrees:

    def __init__(self):
        self.name = 'gbt'

    def train(self, X, y):
        params = {
            'n_estimators': [100, 500],
            'max_depth': [3, 5, 10],
            'subsample': [0.7, 1.0],
            'eta': [0.01, 0.1, 0.3],
            'lambda': [0.0, 1.0, 10.0]
        }
        gbt = GridSearchCV(
            estimator=xgb.XGBRegressor(),
            param_grid=params,
            scoring='r2',
            cv=5,
            #verbose=5,
            n_jobs=1
        )
        gbt.fit(X, y)
        self.model = gbt.best_estimator_

    def predict(self, X):
        y = self.model.predict(X)
        return y

class SupportVectorMachine:

    def __init__(self):
        self.name = 'sv'

    def train(self, X, y):

        self.label_scaler = StandardScaler()
        self.label_scaler.fit(y.reshape(-1,1))
        y_sc = self.label_scaler.transform(y.reshape(-1,1)).reshape(-1)

        params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]}
        sv = GridSearchCV(
            estimator = SVR(
                kernel='rbf',
                gamma='scale'
            ),
            param_grid=params,
            scoring='r2',
            cv=5,
            n_jobs=1
        )
        sv.fit(X, y_sc)
        self.model = sv.best_estimator_

    def predict(self, X):
        y_sc = self.model.predict(X)
        y = self.label_scaler.inverse_transform(y_sc.reshape(-1,1))
        return y.reshape(-1)

class NeuralNetwork:

    def __init__(self):
        self.name = 'nn'

    def train(self, X, y):

        self.label_scaler = StandardScaler()
        self.label_scaler.fit(y.reshape(-1,1))
        y_sc = self.label_scaler.transform(y.reshape(-1,1)).reshape(-1)

        params = {
            'hidden_layer_sizes': [(100,100), (50,50), (10,10)],
            'solver': ['adam', 'lbfgs']
        }
        mlp = GridSearchCV(
            estimator = MLPRegressor(
                early_stopping=True,
                activation='relu',
                validation_fraction=0.1,
                solver='lbfgs',
                max_iter=10000,
                random_state=1
            ),
            param_grid=params,
            scoring='r2',
            cv=5,
            #verbose=5,
            n_jobs=1
        )
        mlp.fit(X, y)
        self.model = mlp.best_estimator_

    def predict(self, X):
        y_sc = self.model.predict(X)
        y = self.label_scaler.inverse_transform(y_sc.reshape(-1,1))
        return y.reshape(-1)

class ExactGPModel(gpytorch.models.ExactGP):

    def __init__(self, train_x, train_y, likelihood, isotropic=False):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        if isotropic:
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()) 
        else: 
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[1])) 

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class GaussianProcess:

    def __init__(self, isotropic):
        self.isotropic = isotropic
        self.name = 'gp' if isotropic else 'gp_ard'
            
    def train(self, X, y, train_iter=10000, print_progress=True):

        self.label_scaler = StandardScaler()
        self.label_scaler.fit(y.reshape(-1,1))
        y_sc = self.label_scaler.transform(y.reshape(-1,1)).reshape(-1)

        X = torch.tensor(X, dtype=torch.float64)
        y_sc = torch.tensor(y_sc, dtype=torch.float64)

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.model = ExactGPModel(X, y_sc, self.likelihood, self.isotropic)
        self.model.double()
        self.likelihood.train()
        self.model.train()

        losses = []
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        for i in range(train_iter):

            optimizer.zero_grad()
            output = self.model(X)
            loss = -mll(output, y_sc.view(-1))
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

            if i > 1000 and np.abs(loss.item() - losses[-100]) / np.abs(losses[-100]) < 1e-3:
                print('Stopping early!')
                break

            # if print_progress and (i+1) % 100 == 0:
            #     print('Iter %d/%d - Loss: %.3f | Lengthscale: %.3f | Noise: %.3f' % (
            #         i + 1, train_iter, loss.item(),
            #         self.model.covar_module.base_kernel.lengthscale.item(),
            #         self.model.likelihood.noise.item()
            #     ))

    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float64)
        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood(self.model(X))
            y_pred = observed_pred.mean.detach().numpy()
            y_pred = self.label_scaler.inverse_transform(y_pred.reshape(-1,1)).reshape(-1)
        return y_pred

class K_NearestNeighbors:
    
    def __init__(self):
        self.name = 'knn'

    def train(self, X, y):

        params = {
            'n_neighbors' : [2, 3, 5, 8, 10, 15],
            'weights' : ['uniform', 'distance'],
            'p' : [1, 2, 3, 5, 10, 15]
        }
        knr = GridSearchCV(
            estimator=KNeighborsRegressor(),
            param_grid=params,
            scoring='r2',
            cv=5,
            n_jobs=1
        )
        knr.fit(X, y)
        self.model = knr.best_estimator_

    def predict(self, X):
        return self.model.predict(X)

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    metrics = [0, 0, 0, 0, 0, 0]
    metrics[0] = r2_score(y_test, y_pred)
    metrics[1] = mean_absolute_error(y_test, y_pred)
    metrics[2] = root_mean_squared_error(y_test, y_pred)
    metrics[3] = (root_mean_squared_error(y_test, y_pred) / np.std(y_test))
    metrics[4] = pearsonr(y_test, y_pred).statistic
    metrics[5] = spearmanr(y_test, y_pred).statistic
    return metrics

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--database')
    parser.add_argument('--target')
    parser.add_argument('--size', type=int)
    parser.add_argument('--strategy')
    parser.add_argument('--sampler')
    parser.add_argument('--model')
    parser.add_argument('--batch')
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    import time
    start = time.perf_counter()

    # Load appropriate dataset.
    np.random.seed(args.seed)
    dataset = np.load(f'./databases/{args.database}-{args.target}.npy')
    X = dataset[:,0:-1]
    y = dataset[:,-1]
    X = StandardScaler().fit_transform(X) # Scale because we know this information beforehand.
    print(f'Shape of dataset: {X.shape}')

    # Get appropriate training data name.
    if args.strategy == 'sf':
        filename = f'{args.database}-{args.target}-sf-{args.sampler}-{args.seed}'
    else:
        filename = f'{args.database}-{args.target}-al-{args.model}-{args.batch}-{args.seed}'

    # Get data indices.
    train_idx = np.load(f'./datasets/size_{args.size}/{filename}.npy')
    X_train = X[train_idx]
    y_train = y[train_idx]

    # Define models.
    models = [
        ['rf', RandomForest()],
        ['gbt', GradientBoostedTrees()],
        ['sv', SupportVectorMachine()],
        ['nn', NeuralNetwork()],
        ['gp', GaussianProcess(isotropic=True)],
        ['gp_ard', GaussianProcess(isotropic=False)],
        ['knn', K_NearestNeighbors()]
    ]

    # Train and score models.
    result = {}
    for model_name, model in models:
        print(f'Training model: {model_name}')
        model.train(X_train, y_train)
        y_pred = model.predict(X)
        result[model_name] = evaluate_model(model, X, y)

    # Write results to file.
    with open('results.csv', 'a') as handle:
        if args.strategy == 'sf':
            write_str = f'{args.database},{args.target},{args.size},sf,{args.sampler},none,none,{args.seed}'
        else:
            write_str = f'{args.database},{args.target},{args.size},al,none,{args.model},{args.batch},{args.seed}'
        for model_name in result.keys():
            for score in result[model_name]:
                write_str = write_str + f',{score:.5f}'
        write_str = write_str + '\n'
        handle.write(write_str)

    end = time.perf_counter()
    print(f'{end-start:.3f} seconds for evaluation.')