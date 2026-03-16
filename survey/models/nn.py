import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

class NN:
    '''
        Scikit-learn implementation of a neural network for guiding
        active learning campaigns. Uses two hidden layers of size 100
        and the L-BFGS optimizer.

        An ensemble of neural networks is trained using different
        random initializations to generate uncertainty computations.
    '''

    def __init__(self, n_models=10):
        self.name = 'nn'
        self.n_models = 10

    def train(self, X, y, tune=False):

        # Scale labels for NN training.
        self.label_scaler = StandardScaler()
        self.label_scaler.fit(y.reshape(-1,1))
        y_sc = self.label_scaler.transform(y.reshape(-1,1)).reshape(-1)

        if tune:
        
            # Train ensemble of NN models.
            self.models = []
            for model_num in range(self.n_models):
                model = MLPRegressor(
                    hidden_layer_sizes=(100,100),
                    solver='lbfgs',
                    max_iter=10000,
                    random_state=model_num,
                    warm_start=True
                )
                model.fit(X,y_sc)
                self.models.append(model)

        else:

            # Fine-tune ensemble of NN models.
            for model_num in range(self.n_models):
                self.models[model_num].fit(X,y_sc)

    def predict(self, X):

        y_samples = []
        for model in self.models:
            y_pred_sc = model.predict(X).reshape(-1,1)
            y_pred = self.label_scaler.inverse_transform(y_pred_sc).reshape(-1)
            y_samples.append(y_pred)

        return np.mean(y_samples, axis=0)

    def get_uncertainties(self, X):
        
        y_samples = []
        for model in self.models:
            y_pred_sc = model.predict(X).reshape(-1,1)
            y_pred = self.label_scaler.inverse_transform(y_pred_sc).reshape(-1)
            y_samples.append(y_pred)

        return np.std(y_samples, axis=0)