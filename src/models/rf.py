import numpy as np
from sklearn.ensemble import RandomForestRegressor

class RF:
    '''
        Scikit-learn implementation of a random forest for guiding
        active learning campaigns. Uses 100 estimators and 1/2 of 
        the available features.
    '''

    def __init__(self):
        self.name = 'rf'

    def train(self, X, y, tune=False):

        self.model = RandomForestRegressor(
            n_estimators=100,
            max_features=0.5,
            random_state=1
        )
        self.model.fit(X,y)

    def predict(self, X):
        
        return self.model.predict(X)

    def get_uncertainties(self, X):
        
        y_samples = [tree.predict(X) for tree in self.model.estimators_]

        return np.std(y_samples, axis=0)