# from ibug import IBUGWrapper
import numpy as np
import xgboost as xgb

class GBT:
    '''
        XGBoost implementation of gradient boosted decision trees for 
        guiding active learning campaigns. Uncertainties are computed
        using the method of J. Brophy and D. Lowd in NeurIPS (2022).
    '''

    def __init__(self):
        self.name = 'gbt'

    def train(self, X, y, tune=False):
        
        self.model = xgb.XGBRegressor(
            n_estimators=1000, 
            learning_rate=0.25,
            reg_lambda=0.01,
            reg_alpha=0.1,
            subsample=0.85,
            colsample_bytree=0.3,
            colsample_bylevel=0.5
        )
        self.model.fit(X,y)
        self.X_train = X
        self.y_train = y

    def predict(self, X):

        return self.model.predict(X)

    # def get_uncertainties(self, X):

    #     ibug_model = IBUGWrapper().fit(self.model, self.X_train, self.y_train)
    #     ibug_model.set_tree_subsampling(1.0, 'random')
    #     _, y_var = ibug_model.pred_dist(X)

    #     return np.sqrt(y_var)