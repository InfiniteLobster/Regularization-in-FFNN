#--------------------Libraries--------------------#
import numpy as np
#--------------------Functions--------------------#
#Here are the activation functions and their derivatives used by model layers
#sigmopid activation function
def sigmoid(z: np.ndarray) -> np.ndarray:
    #creating array for storing sigmoid values
    values = np.asarray(z, dtype=float)
    sig = np.empty_like(values)
    #getting boolean array for values >= 0 for numerical stability in sigmoid calculation
    positive = values >= 0
    #calculating sigmoid values for positive values using stable formula
    sig[positive] = 1 / (1 + np.exp(-values[positive]))
    #calculating sigmoid values for negative values using stable formula
    exp_values = np.exp(values[~positive])
    sig[~positive] = exp_values / (1 + exp_values)
    #returning output
    return sig
#derivative of sigmoid activation function
def sigmoid_derivative(a: np.ndarray) -> np.ndarray:
    #calculating sigmoid values for input array
    sig_deriv = a * (1 - a)
    #returning output
    return sig_deriv
#Relu activation function
def relu(z: np.ndarray) -> np.ndarray:
    #creating array for storing Relu values
    rel = np.maximum(0, z)
    #returning output
    return rel
#derivative of Relu activation function
def relu_derivative(a: np.ndarray) -> np.ndarray:
    #getting derivative values for Relu. For stability zero values are considered as non-active, i.e. derivative is 0 for them.
    rel_deriv = np.where(a > 0, 
                         1, 
                         0)
    #returning output
    return rel_deriv