import random

import numpy as np


def mean_squared_error_gd(
    y: np.ndarray,
    tx: np.ndarray,
    initial_w: np.ndarray,
    max_iters: int,
    gamma: float,
    track_history: bool = False,
) -> tuple[np.ndarray, float | list[float]]:
    w = initial_w
    loss_history = []
    for _ in range(max_iters):
        grad = grad_mse(y, tx, w)
        w = grad_step(w, gamma, grad)
        if track_history:
            loss_history.append(mse(y, tx, w))

    loss = mse(y, tx, w)
    return w, loss_history if track_history else loss


def mean_squared_error_sgd(
    y: np.ndarray,
    tx: np.ndarray,
    initial_w: np.ndarray,
    max_iters: int,
    gamma: float,
    track_history: bool = False,
) -> tuple[np.ndarray, float | list[float]]:

    w = initial_w
    loss_history = []
    N = tx.shape[0]
    for _ in range(max_iters):
        n = random.randint(0, N - 1)
        y_n = y[n]
        x_n = tx[n, :]
        grad = grad_s_mse(y_n, x_n, w)
        w = grad_step(w, gamma, grad)
        if track_history:
            loss_history.append(mse(y, tx, w))

    loss = mse(y, tx, w)
    return w, loss_history if track_history else loss


def least_squares(y: np.ndarray, tx: np.ndarray) -> tuple[np.ndarray, float]:
    """Calculate the least squares solution.
       returns mse, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.

    """
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    loss = mse(y, tx, w)
    return w, loss


def ridge_regression(y: np.ndarray, tx: np.ndarray, lambda_: float) -> tuple[np.ndarray, float]:
    """implement ridge regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss:
    """
    D = tx.shape[1]
    N = tx.shape[0]
    A = (tx.T @ tx) + (np.eye(D) * 2 * lambda_ * N)
    b = tx.T @ y
    weights = np.linalg.solve(A, b)
    loss = mse(y, tx, weights)
    return weights, loss


def logistic_regression(
    y: np.ndarray,
    tx: np.ndarray,
    initial_w: np.ndarray,
    max_iters: int,
    gamma: float,
    track_history: bool = False,
) -> tuple[np.ndarray, float | list[float]]:

    w = initial_w
    loss_history = []
    for _ in range(max_iters):
        grad = grad_nll(y, tx, w)
        w = grad_step(w, gamma, grad)
        if track_history:
            loss_history.append(nll(y, tx, w))

    loss = nll(y, tx, w)
    return w, loss_history if track_history else loss


def reg_logistic_regression(
    y: np.ndarray,
    tx: np.ndarray,
    lambda_: float,
    initial_w: np.ndarray,
    max_iters: int,
    gamma: float,
    track_history: bool = False,
) -> tuple[np.ndarray, float | list[float]]:

    w = initial_w
    loss_history = []
    for _ in range(max_iters):
        grad = grad_nll(y, tx, w)
        grad += 2 * lambda_ * w
        w = grad_step(w, gamma, grad)
        if track_history:
            loss_history.append(nll(y, tx, w))

    loss = nll(y, tx, w)
    return w, loss_history if track_history else loss


# ==== related functions
def grad_step(w: np.ndarray, gamma: float, grad: np.ndarray) -> np.ndarray:
    """
    Do one step of gradient descent using logistic regression. Return the loss and the updated w.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
        gamma: float

    Returns:
        loss: scalar number
        w: shape=(D, 1)
    """
    next_w = w - gamma * grad
    return next_w


# ==== linear regression / ridge regression related funcitons
def mse(y: np.ndarray, tx: np.ndarray, w: np.ndarray) -> float:
    """Calculate MSE

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """

    e = y - tx @ w
    N = len(y)
    return float((1 / 2) * (1 / N) * e @ e)


def grad_mse(y: np.ndarray, tx: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Computes the gradient at w.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2, ). The vector of model parameters.

    Returns:
        An numpy array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """
    e = y - tx @ w
    # if isinstance(y, np.ndarray):
    N = y.shape[0]
    # else:
        # N = 1
    return -1 / N * tx.T @ e

def grad_s_mse(y: float, tx: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Computes the gradient at w for a single sample.

    Args:
        y: scalar
        tx: numpy array of shape=(2, )
        w: numpy array of shape=(2, ). The vector of model parameters.
    """
    e = y - tx @ w
    return -1 * tx.T * e


# ==== Logistic regression helpers
def nll(y: np.ndarray, tx: np.ndarray, w: np.ndarray) -> float:
    """compute the cost by negative log likelihood.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a non-negative loss
    """

    z = tx @ w
    return float(np.mean(np.logaddexp(0, z) - y * z))


def grad_nll(y: np.ndarray, tx: np.ndarray, w: np.ndarray) -> np.ndarray:
    """compute the gradient of loss.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a vector of shape (D, 1)
    """
    N = tx.shape[0]
    residual = sigmoid(tx @ w) - y
    gradient = tx.transpose_dot(residual) if hasattr(tx, "transpose_dot") else tx.T @ residual
    return gradient / N


def sigmoid(t: np.ndarray | float) -> np.ndarray | float:
    """apply sigmoid function on t.

    Args:
        t: scalar or numpy array

    Returns:
        scalar or numpy array

    """
    return np.exp(-np.logaddexp(0, -t))
