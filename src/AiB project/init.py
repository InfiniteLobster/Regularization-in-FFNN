#--------------------Libraries--------------------#
import numpy as np
#--------------------Functions--------------------#
#By design each function is for one weight layer initialization
#Initialization of weights and biases as zeros
def init_zero(shape: np.ndarray, dtype: type) -> np.ndarray:
    #initializing weight matrix (with zeros as it is zero initialization) with given shape and data type
    weights = np.zeros(shape, dtype)
    #returning output
    return weights
#Xavier/Glorot uniform initialization of weights
def init_xavier_uniform(shape: np.ndarray, dtype: type) -> np.ndarray:
    #calculating fan_in and fan_out for Xavier/Glorot initialization
    fan_in = shape[1]
    fan_out = shape[0]
    # Xavier/Glorot uniform initialization
    limit = np.sqrt(6 / (fan_in + fan_out))
    weights = np.random.uniform(-limit, limit, size=shape).astype(dtype)
    #returning output
    return weights
