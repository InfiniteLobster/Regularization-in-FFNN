#--------------------Libraries--------------------#
import numpy as np
#--------------------Functions--------------------#
#
#
def sigmoid(a: np.ndarray) -> np.ndarray:
    # 
    sig = np.where(a >= 0,
                    1 / (1 + np.exp(-a)),
                    np.exp(a) / (1 + np.exp(a))
                    )
    #
    return sig
def sigmoid_derivative(a: np.ndarray) -> np.ndarray:
    sig = sigmoid(a)
    #
    sig_deriv = sig * (1 - sig)
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