#--------------------Libraries--------------------#
import numpy as np
#------------------Py files code------------------#
from model import FFNN
from activ import sigmoid, sigmoid_derivative, relu, relu_derivative #tanh, tanh_derivative
from init import init_zero, init_xavier_uniform
from loss import mse_loss, mse_loss_derivative
from update import standard
#--------------------Functions--------------------#
#
def train_GD(model_totrain: FFNN, X:np.ndarray, Y:np.ndarray,update_func: callable = standard, learning_rate: float = 0.01, epochs: int = 1000) -> FFNN:
    #
    loss_log = []
    for epoch in range(epochs):
        #forward
        out = model_totrain.forward(X)
        #loss
        loss = mse_loss(Y, out[-1])
        if(epoch % 100 == 0):
            loss_log.append(loss)
        #loss der
        loss_der = mse_loss_derivative(Y, out[-1])
        #backward(update)
        model_totrain.backward(A = out, loss_derivative = loss_der, update_func = update_func, learning_rate = learning_rate)
    #
    return model_totrain, loss_log