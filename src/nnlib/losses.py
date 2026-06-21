import numpy as np


def square_loss(y_pred, y_target):
    return (y_pred - y_target) ** 2


def square_loss_derivative(y_pred, y_target):
    return 2 * (y_pred - y_target)


def categorical_crossentropy(y_pred, y_target):
    y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.sum(y_target * np.log(y_pred_clipped), axis=1, keepdims=True)


def categorical_crossentropy_derivative(y_pred, y_target):
    return y_pred - y_target


def binary_crossentropy(y_pred, y_target):
    y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -(y_target * np.log(y_pred_clipped) + (1 - y_target) * np.log(1 - y_pred_clipped))


def binary_crossentropy_derivative(y_pred, y_target):
    y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -(y_target / y_pred_clipped - (1 - y_target) / (1 - y_pred_clipped))
