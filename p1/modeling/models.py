
from abc import ABC, abstractmethod

import numpy as np

from p1.modeling.train import (
    least_squares,
    logistic_regression,
    mean_squared_error_gd,
    mean_squared_error_sgd,
    reg_logistic_regression,
    ridge_regression,
    sigmoid,
)


class Model(ABC):
    w: np.ndarray | None
    train_loss: float | list[float] | None

    def __init__(self) -> None:
        self.w = None
        self.train_loss = None

    @abstractmethod
    def fit(self, y: np.ndarray, tx: np.ndarray) -> None:
        """Fit the model and retain its training loss."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, tx: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _fitted_weights(self) -> np.ndarray:
        if self.w is None:
            raise RuntimeError("Model must be fitted before calling predict.")
        return self.w


class LinearRegressLS(Model):
    def __init__(self):
        super().__init__()

    def fit(self, y: np.ndarray, tx: np.ndarray) -> None:
        self.w, self.train_loss = least_squares(y, tx)

    def predict(self, tx: np.ndarray) -> np.ndarray:
        return tx @ self._fitted_weights()

class LinearRegressGD(Model):
    def __init__(self, max_iters=1000, gamma=0.01, track_history=False):
        super().__init__()
        self.max_iters = max_iters
        self.gamma = gamma
        self.track_history = track_history

    def fit(self, y: np.ndarray, tx: np.ndarray) -> None:
        initial_w = np.zeros(tx.shape[1])
        self.w, self.train_loss = mean_squared_error_gd(
            y, tx, initial_w, self.max_iters, self.gamma, self.track_history
        )

    def predict(self, tx: np.ndarray) -> np.ndarray:
        return tx @ self._fitted_weights()

class LinearRegressSGD(Model):
    def __init__(self, max_iters=1000, gamma=0.01, track_history=False):
        super().__init__()
        self.max_iters = max_iters
        self.gamma = gamma
        self.track_history = track_history

    def fit(self, y: np.ndarray, tx: np.ndarray) -> None:
        initial_w = np.zeros(tx.shape[1])
        self.w, self.train_loss = mean_squared_error_sgd(
            y, tx, initial_w, self.max_iters, self.gamma, self.track_history
        )

    def predict(self, tx: np.ndarray) -> np.ndarray:
        return tx @ self._fitted_weights()
    
class RidgeRegress(Model):
    def __init__(self, lambda_):
        super().__init__()
        self.lambda_ = lambda_

    def fit(self, y: np.ndarray, tx: np.ndarray) -> None:
        self.w, self.train_loss = ridge_regression(y, tx, self.lambda_)

    def predict(self, tx: np.ndarray) -> np.ndarray:
        return tx @ self._fitted_weights()


class LogisticRegress(Model):
    def __init__(self, max_iters, gamma, track_history=False):
        super().__init__()
        self.max_iters = max_iters
        self.gamma = gamma
        self.track_history = track_history
    
    def fit(self, y: np.ndarray, tx: np.ndarray) -> None:
        initial_w = np.zeros(tx.shape[1])
        self.w, self.train_loss = logistic_regression(
            y, tx, initial_w, self.max_iters, self.gamma, self.track_history
        )
    
    def predict(self, tx: np.ndarray) -> np.ndarray:
        return sigmoid(tx @ self._fitted_weights())


class RegLogisticRegress(Model):
    def __init__(self, lambda_, max_iters, gamma, track_history=False):
        super().__init__()
        self.lambda_ = lambda_
        self.max_iters = max_iters
        self.gamma = gamma
        self.track_history = track_history
    
    def fit(self, y: np.ndarray, tx: np.ndarray) -> None:
        initial_w = np.zeros(tx.shape[1])
        self.w, self.train_loss = reg_logistic_regression(
            y,
            tx,
            self.lambda_,
            initial_w,
            self.max_iters,
            self.gamma,
            self.track_history,
        )
    
    def predict(self, tx: np.ndarray) -> np.ndarray:
        return sigmoid(tx @ self._fitted_weights())
