"""The six required machine-learning method implementations for Project 1."""

import random

import numpy as np


def mean_squared_error_gd(
    y: np.ndarray,
    tx: np.ndarray,
    initial_w: np.ndarray,
    max_iters: int,
    gamma: float,
) -> tuple[np.ndarray, float]:
    """Run batch gradient descent for linear regression."""
    w = initial_w
    for _ in range(max_iters):
        grad = grad_mse(y, tx, w)
        w = grad_step(w, gamma, grad)

    return w, mse(y, tx, w)


def mean_squared_error_sgd(
    y: np.ndarray,
    tx: np.ndarray,
    initial_w: np.ndarray,
    max_iters: int,
    gamma: float,
) -> tuple[np.ndarray, float]:
    """Run mini-batch-size-one stochastic gradient descent for linear regression."""
    w = initial_w
    N = tx.shape[0]
    for _ in range(max_iters):
        n = random.randint(0, N - 1)
        y_n = y[n]
        x_n = tx[n, :]
        grad = grad_s_mse(y_n, x_n, w)
        w = grad_step(w, gamma, grad)

    return w, mse(y, tx, w)


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
    return w, mse(y, tx, w)


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
    A = tx.T @ tx + np.eye(D) * 2 * lambda_ * N
    weights = np.linalg.solve(A, tx.T @ y)
    return weights, mse(y, tx, weights)


def logistic_regression(
    y: np.ndarray,
    tx: np.ndarray,
    initial_w: np.ndarray,
    max_iters: int,
    gamma: float,
) -> tuple[np.ndarray, float]:
    """Run gradient descent for logistic regression with labels in {0, 1}."""
    w = initial_w
    for _ in range(max_iters):
        grad = grad_nll(y, tx, w)
        w = grad_step(w, gamma, grad)

    return w, nll(y, tx, w)


def reg_logistic_regression(
    y: np.ndarray,
    tx: np.ndarray,
    lambda_: float,
    initial_w: np.ndarray,
    max_iters: int,
    gamma: float,
) -> tuple[np.ndarray, float]:
    """Run L2-regularized logistic regression gradient descent."""
    w = initial_w
    for _ in range(max_iters):
        grad = grad_nll(y, tx, w)
        grad += 2 * lambda_ * w
        w = grad_step(w, gamma, grad)

    return w, nll(y, tx, w)


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
    N = y.shape[0]
    return -1 / N * tx.T @ e


def grad_s_mse(y: float, tx: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Computes the gradient at w for a single sample.

    Args:
        y: scalar
        tx: numpy array of shape=(2, )
        w: numpy array of shape=(2, ). The vector of model parameters.
    """
    e = y - tx @ w
    return -1 * tx * e


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
    return tx.T @ residual / N


def sigmoid(t: np.ndarray | float) -> np.ndarray | float:
    """apply sigmoid function on t.

    Args:
        t: scalar or numpy array

    Returns:
        scalar or numpy array

    """
    return np.exp(-np.logaddexp(0, -t))
