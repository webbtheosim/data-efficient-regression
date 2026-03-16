import numpy as np
from scipy.spatial.distance import cdist

from samplers import sample

def topk(batch_size, chosen_idx, y_std):
    '''
        Chooses the `batch_size` most uncertain points.
    '''
    
    y_std[chosen_idx] = -np.inf
    batch_idx = np.argsort(-y_std).reshape(-1)[0:batch_size]

    return batch_idx.tolist()

def cluster_margin(batch_size, X, chosen_idx, y_std, chosen_feat=None):
    '''
        Selects points according to a method inspired by the cluster-margin 
        algorithm specified in Citovsky et al. (2022). The k_m most uncertain
        points are chosen, and batch_size points are selected by performing
        k-medoids on the chosen points, ensuring a diverse sample of uncertain
        points are selected.

        This method differs from Citovsky et al. (2022) in that (1) we are not
        representing points with neural network embeddings and (2) we are re-
        placing randomly choosing points from k-means generated samples by just
        running the k-medoids algorithm. The former is because we are not always
        using neural networks as our active learning guide. The latter is because
        k-medoids more simply achieves the same goal and is viable for our batch
        sizes (unlike the 100k+ batch sizes considered in their work).
    '''

    y_std[chosen_idx] = -np.inf 
    k_m = int(0.1 * X.shape[0]) # 10% of the design space is considered.
    candidate_idx = np.argsort(-y_std).reshape(-1)[0:k_m]
    X_candidate = X[candidate_idx]

    if chosen_feat is not None:
        X_candidate = X_candidate[:,chosen_feat]

    subsample_idx = sample('medoids', X_candidate, batch_size, seed=1)
    batch_idx = [candidate_idx[idx] for idx in subsample_idx]

    return batch_idx

def hallucinate(batch_size, X, y, chosen_idx, surrogate, chosen_feat=None):
    '''
        Greedily chooses a batch of points by selecting the most
        uncertain point, retraining the model assuming its prediction
        at that point is correct (i.e., hallucinating), recomputing
        uncertainties, and repeating this process until a sufficient
        number of points are chosen. Also called "kriging believer" per
        Ginsbourger et al. (2007).
    '''

    if chosen_feat is not None:
        X = X[:,chosen_feat]
        
    idx = [i for i in chosen_idx]
    y_train = y[chosen_idx]
    for step in range(batch_size):
        print(f'Selecting point: {step + 1} / {batch_size}...')
        
        y_std = surrogate.get_uncertainties(X)
        y_std = y_std + 1e-30 * np.random.random(size=y_std.shape[0])
        y_std[idx] = -np.inf
        
        new_idx = np.argmax(y_std).item()
        idx.append(new_idx)
        y_new = surrogate.predict(X[new_idx].reshape(1,-1))
        y_train = np.hstack((y_train, y_new))

        surrogate.train(X[idx], y_train, tune=False)

    return idx[-batch_size:]

def multiobjective(batch_size, X, chosen_idx, y_std, chosen_feat=None):
    '''
        Recognizing that batch selection is a multiobjective optimization
        between (1) model-predicted uncertainties and (2) diversity in
        feature space, we choose a batch of points by sequentially optimizing`
        a multiobjective acquisition function that varies the prioritization
        of model uncertainties and distance from previously chosen points.
    '''

    idx = [i for i in chosen_idx]
    weights = np.linspace(0, 1, num=batch_size)
    y_std_sc = y_std / np.max(y_std)

    for w in weights:

        # Compute distances to already chosen points.
        if chosen_feat is not None:
            X_select = X[idx]
            d = cdist(X[:,chosen_feat], X_select[:,chosen_feat])
        else:
            d = cdist(X, X[idx])
        d = np.min(d, axis=1)
        d_sc = d / np.max(d)

        # Vectorize objective function.
        acq = (1 - w) * y_std_sc + w * d_sc
        new_idx = np.argmax(acq)
        idx.append(new_idx)

    return idx[-batch_size:]

def select_batch(
        batch_strat, batch_size, X, y, chosen_idx, surrogate, y_std, chosen_feat=None
    ):
    '''
        Wrapper function for choosing batches of points according to
        the specified strategy.
    '''

    if batch_strat == 'topk':
        batch_idx = topk(batch_size, chosen_idx, y_std)
    if batch_strat == 'cluster_margin':
        batch_idx = cluster_margin(batch_size, X, chosen_idx, y_std, chosen_feat)
    if batch_strat == 'hallucinate':
        batch_idx = hallucinate(batch_size, X, y, chosen_idx, surrogate, chosen_feat)
    if batch_strat == 'pareto':
        batch_idx = multiobjective(batch_size, X, chosen_idx, y_std, chosen_feat)

    return batch_idx
