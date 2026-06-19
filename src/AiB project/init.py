#--------------------Libraries--------------------#
import numpy as np
#--------------------Functions--------------------#
#By design each function is for one weight layer initialization
#
def init_zero(shape: np.ndarray, dtype: type) -> np.ndarray:
    #
    weights = np.zeros(shape, dtype)
    #
    return weights
#
def init_xavier_uniform(shape: np.ndarray, dtype: type) -> np.ndarray:
    #
    fan_in = shape[1]
    fan_out = shape[0]
    # Xavier/Glorot uniform initialization
    limit = np.sqrt(6 / (fan_in + fan_out))
    weights = np.random.uniform(-limit, limit, size=shape).astype(dtype)
    #
    return weights
