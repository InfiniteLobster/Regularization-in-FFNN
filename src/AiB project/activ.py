#--------------------Libraries--------------------#
import numpy as np
#--------------------Functions--------------------#
#
#
def sigmoid(a: np.ndarray) -> np.ndarray:
    #
    values = np.asarray(a, dtype=float)
    sig = np.empty_like(values)
    positive = values >= 0

    sig[positive] = 1 / (1 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    sig[~positive] = exp_values / (1 + exp_values)
    #
    return sig
def sigmoid_derivative(a: np.ndarray) -> np.ndarray:
    sig_deriv = a * (1 - a)
    #
    return sig_deriv
#
def relu(a: np.ndarray) -> np.ndarray:
    #
    rel = np.maximum(0, a)
    #
    return rel
def relu_derivative(a: np.ndarray) -> np.ndarray:
    #
    rel_deriv = np.where(a > 0, 
                         1, 
                         0)
    #
    return rel_deriv