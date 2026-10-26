from collections import defaultdict

import numpy as np

from p1.metrics import accuracy, f1, precision, recall
from p1.modeling.models import Model


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold.

    Args:
        y:      shape=(N,)
        k_fold: K in K-fold, i.e. the fold num
        seed:   the random seed

    Returns:
        A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold

    >>> build_k_indices(np.array([1., 2., 3., 4.]), 2, 1)
    array([[3, 2],
           [0, 1]])
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


def get_closest(x):
    """If x is closer to 1, return one. Otherwise zero"""
    if abs(x) >= 0.5:
        return 1
    return 0
gc = np.vectorize(get_closest)



def cross_validation(model: Model, y, x, k_indices, k):
    """return the loss of ridge regression for a fold corresponding to k_indices

    Args:
        y:          shape=(N,)
        x:          shape=(N,)
        k_indices:  2D array returned by build_k_indices()
        k:          scalar, the k-th fold (N.B.: not to confused with k_fold which is the fold nums)
        lambda_:    scalar, cf. ridge_regression()
        degree:     scalar, cf. build_poly()

    Returns:
        train and test root mean square errors rmse = sqrt(2 mse)

    >>> cross_validation(np.array([1.,2.,3.,4.]), np.array([6.,7.,8.,9.]), np.array([[3,2], [0,1]]), 1, 2, 3)
    (0.019866645527596753, 0.33555914361295197)
    """

    ### SOLUTION
    # get k'th subgroup in test, others in train
    te_indice = k_indices[k]
    tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    tr_indice = tr_indice.reshape(-1)
    y_te = y[te_indice]
    y_tr = y[tr_indice]
    x_te = x[te_indice]
    x_tr = x[tr_indice]
    # form data with polynomial degree
    # w = ridge_regression(y_tr, tx_tr, lambda_)
    # model = model_class() 

    model.fit(y_tr, x_tr)

    preds = model.predict(x_te)
    preds = get_closest(preds)

    # calculate the loss for train and test data
    return {
        'accuracy': accuracy(y_te, preds),
        'recall': recall(y_te, preds),
        'precision': precision(y_te, preds),
        'f1': f1(y_te, preds)
    }

def cross_eval_model(model: Model, y, tx, k_fold, seed):
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data

    metrics = defaultdict(list)


    for k in range(k_fold):
        # model = model_class() 
        k_metrics = cross_validation(model, y, tx, k_indices, k)
        for key, val in k_metrics:
            metrics[key].append(val)
    avg_metrics = {key: np.mean(vals) for key, vals in metrics} 

    return avg_metrics
