import numpy as np
from sklearn.neighbors import KNeighborsRegressor

class KNN:
    '''
        Scikit-learn implementation of a k-neighbors regressor for 
        guiding active learning campaigns. Computes uncertainty using
        the standard deviation of neighbor labels. k is set to 5.
    '''
    
    def __init__(self):
        self.name = 'knn'

    def train(self, X, y, tune=False):

        self.model = KNeighborsRegressor(n_neighbors=5)
        self.model.fit(X,y)
        self.y_train = y

    def predict(self, X):

        return self.model.predict(X)

    def get_uncertainties(self, X):

        neighbors_idx = self.model.kneighbors(X, return_distance=False)
        y_std = np.std(self.y_train[neighbors_idx], axis=1)
        
        return y_std