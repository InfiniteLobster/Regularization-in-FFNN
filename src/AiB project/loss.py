#--------------------Libraries--------------------#
import numpy as np
#--------------------Functions--------------------#
#
#
def mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    #
    loss = np.mean((y_true - y_pred) ** 2)
    #
    return loss
def mse_loss_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    #
    loss_deriv = 2 * (y_pred - y_true) / y_true.size
    #
    return loss_deriv
    
