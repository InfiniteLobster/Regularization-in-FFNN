#--------------------Libraries--------------------#
import numpy as np
#--------------------Functions--------------------#
#Those are the functions for updating the weights/biases based on gradients, i.e. what is -= in standard interpretation
#weight and bias updayes without regularization
def standard(grad: np.ndarray, learning_rate: float, weights_array: np.ndarray, lambda_: float = 0) -> np.ndarray:
    #update
    updated_grad = learning_rate * grad
    #returning output 
    return updated_grad
#weight and bias updates with L1 regularization
def reg_L1(grad: np.ndarray, learning_rate: float, weights_array: np.ndarray, lambda_: float) -> np.ndarray:
    #update
    updated_grad = learning_rate * (grad  + lambda_ * np.sign(weights_array))
    #returning output 
    return updated_grad
#weight and bias updates with L2 regularization
def reg_L2(grad: np.ndarray, learning_rate: float, weights_array: np.ndarray, lambda_: float) -> np.ndarray:
    #update
    updated_grad = learning_rate * (grad  + lambda_ * weights_array)
    #returning output 
    return updated_grad