import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the accuracy of predictions.

    Args:
        y_true: shape=(N, 1)
        y_pred: shape=(N, 1)

    Returns:
        accuracy: float
    """
    if len(y_true) == 0:
        return 0.0
    correct_predictions = np.sum(y_true == y_pred)
    accuracy = correct_predictions / len(y_true)
    return accuracy

def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the recall of predictions.

    Args:
        y_true: shape=(N, 1)
        y_pred: shape=(N, 1)

    Returns:
        recall: float
    """
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    false_negatives = np.sum((y_true == 1) & (y_pred == 0))
    if true_positives + false_negatives == 0:
        return 0.0
    recall = true_positives / (true_positives + false_negatives)
    return recall

def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the precision of predictions.

    Args:
        y_true: shape=(N, 1)
        y_pred: shape=(N, 1)

    Returns:
        precision: float
    """
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    false_positives = np.sum((y_true == 0) & (y_pred == 1))
    if true_positives + false_positives == 0:
        return 0.0
    precision = true_positives / (true_positives + false_positives)
    return precision

def f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the F1 score of predictions.

    Args:
        y_true: shape=(N, 1)
        y_pred: shape=(N, 1)

    Returns:
        f1_score: float
    """
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    if prec + rec == 0:
        return 0.0
    f1_score = 2 * (prec * rec) / (prec + rec)
    return f1_score
