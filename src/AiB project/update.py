#--------------------Libraries--------------------#
import numpy as np
#--------------------Functions--------------------#
#
#
def standard(grad: np.ndarray, learning_rate: float) -> np.ndarray:
    #
    updated_grad = grad * learning_rate
    #
    return updated_grad
#
def reg_L1(grad: np.ndarray, weights: np.ndarray, learning_rate: float, lambda_: float) -> np.ndarray:
    #
    updated_grad = grad * learning_rate + lambda_ * np.sign(weights)
    #
    return updated_grad
#
def reg_L2(grad: np.ndarray, weights: np.ndarray, learning_rate: float, lambda_: float) -> np.ndarray:
    #
    updated_grad = grad * learning_rate + lambda_ * weights
    #
    return updated_grad