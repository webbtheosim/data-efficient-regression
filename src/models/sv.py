import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

class SV:
    '''
        Scikit-learn implementation of a support vector machine for 
        guiding active learning campaigns. Uses bootstrap aggregation
        with a subsampling fraction of 0.7. All SVMs use an RBF kernel
        with a 'scaled' RBF coefficient and a regularization parameter
        of 1.0.
    '''

    def __init__(self, n_models=30):
        self.name = 'sv'
        self.n_models = 30

    def train(self, X, y, tune=False):

        # Scale labels for SVM training.
        self.label_scaler = StandardScaler()
        self.label_scaler.fit(y.reshape(-1,1))
        y_sc = self.label_scaler.transform(y.reshape(-1,1)).reshape(-1)

        # Train single model for predictions.
        params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]}
        sv = GridSearchCV(
            estimator = SVR(
                kernel='rbf',
                gamma='scale'
            ),
            param_grid=params,
            scoring='r2',
            cv=5
        )
        sv.fit(X, y_sc)
        self.model = sv.best_estimator_
        self.config = sv.best_params_

        # Train ensemble of SVMs.
        self.models = []
        indices = [idx for idx in range(X.shape[0])]
        for _ in range(self.n_models):
            train_idx = np.random.choice(indices, size=int(0.7 * X.shape[0]))
            model = SVR(**self.config)
            model.fit(X[train_idx], y_sc[train_idx])
            self.models.append(model)

    def predict(self, X):

        y_pred_sc = self.model.predict(X).reshape(-1,1)
        y_pred = self.label_scaler.inverse_transform(y_pred_sc).reshape(-1)

        return y_pred

    def get_uncertainties(self, X):
        
        y_samples = []
        for model in self.models:
            y_pred_sc = model.predict(X).reshape(-1,1)
            y_pred = self.label_scaler.inverse_transform(y_pred_sc).reshape(-1)
            y_samples.append(y_pred)

        return np.std(y_samples, axis=0)