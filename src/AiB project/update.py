#--------------------Libraries--------------------#
import numpy as np
#--------------------Functions--------------------#
#Those are the functions for updating the weights/biases based on gradients, i.e. what is -= in standard interpretation
#weight and bias updayes without regularization
def standard(grad: np.ndarray, learning_rate: float) -> np.ndarray:
    #update
    updated_grad = grad * learning_rate
    #returning output 
    return updated_grad
#weight and bias updates with L1 regularization
def reg_L1(grad: np.ndarray, weights: np.ndarray, learning_rate: float, lambda_: float) -> np.ndarray:
    #update
    updated_grad = grad * learning_rate + lambda_ * np.sign(weights)
    #returning output 
    return updated_grad
#weight and bias updates with L2 regularization
def reg_L2(grad: np.ndarray, weights: np.ndarray, learning_rate: float, lambda_: float) -> np.ndarray:
    #update
    updated_grad = grad * learning_rate + lambda_ * weights
    #returning output 
    return updated_grad