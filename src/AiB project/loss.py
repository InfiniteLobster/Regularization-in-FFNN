#--------------------Libraries--------------------#
import numpy as np
#--------------------Functions--------------------#
#Here are the loss functions and their derivatives
#Mean Squared Error loss function
def mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    #calculating MSE
    loss = np.mean((y_true - y_pred) ** 2)
    #returning output
    return loss
#derivative of Mean Squared Error loss function
def mse_loss_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    #calculating derivative of MSE
    loss_deriv = 2 * (y_pred - y_true) / y_true.size
    #returning output
    return loss_deriv
    
