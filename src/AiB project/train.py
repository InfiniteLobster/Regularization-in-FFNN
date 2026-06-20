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
def train_GD(model_totrain: FFNN, X:np.ndarray, Y:np.ndarray,update_func: callable = standard, learning_rate: float = 0.01, epochs: int = 1000, log_freq: int = 100) -> FFNN:
    #declaring variable to store loss values for logging purposes
    loss_log = []
    #iterating through epochs for training
    for epoch in range(epochs):
        #forward
        out = model_totrain.forward(X)
        #loss
        if(epoch % log_freq == 0):
            loss_train = mse_loss(Y, out[-1])
            loss_log.append(loss_train)
        #loss der
        loss_der = mse_loss_derivative(Y, out[-1])
        #backward(update)
        model_totrain.backward(A = out, loss_derivative = loss_der, update_func = update_func, learning_rate = learning_rate)
    #returning output
    return model_totrain, loss_log
#
def train_SGD(model_totrain: FFNN, X:np.ndarray, Y:np.ndarray, batch_size: int = 32, update_func: callable = standard, learning_rate: float = 0.01, epochs: int = 1000, log_freq: int = 100) -> FFNN:
    #declaring variable to store loss values for logging purposes
    loss_log = []
    #getting number of samples for shuffling and batching
    num_samples = X.shape[0]
    #iterating through epochs for training
    for epoch in range(epochs):
        #shuffling data at the beginning of each epoch
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        X_shuffled = X[indices]
        Y_shuffled = Y[indices]
        #iterating through batches
        for i in range(0, num_samples, batch_size):
            #getting current batch of data
            X_batch = X_shuffled[i:i+batch_size]
            Y_batch = Y_shuffled[i:i+batch_size]
            #forward
            out = model_totrain.forward(X_batch)
            #loss
            if(epoch % log_freq == 0 and i == 0): 
                loss_train = mse_loss(Y_batch, out[-1])
                loss_log.append(loss_train)
            #loss der
            loss_der = mse_loss_derivative(Y_batch, out[-1])
            #backward(update)
            model_totrain.backward(A = out, loss_derivative = loss_der, update_func = update_func, learning_rate = learning_rate)
    #returning output
    return model_totrain, loss_log