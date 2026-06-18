#--------------------Libraries--------------------#
import numpy as np
#--------------------Functions--------------------#
#By design each function is for one weight layer initialization
#
def init_zero(shape: np.ndarray, dtype: type) -> None:
    #
    weights = np.zeros(shape, dtype)
    #
    return weights
#
def init_xavier_uniform() -> None:
    raise NotImplementedError("Implement inference")