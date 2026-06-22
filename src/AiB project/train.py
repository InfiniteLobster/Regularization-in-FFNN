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
def train_GD(model: FFNN, X_train:np.ndarray, Y_train:np.ndarray,X_test:np.ndarray, Y_test:np.ndarray, reg_layer:list, loss_info: list = [mse_loss, mse_loss_derivative],lambda_: float = 0, learning_rate: float = 0.01, epochs: int = 1000, log_freq: int = 100) -> FFNN:
    #getting loss function and its derivative
    loss_func = loss_info[0]
    loss_derivative = loss_info[1]
    #declaring variable to store loss values for logging purposes
    loss_train = []
    loss_test = []
    weights_list = []
    biases_list = []
    #iterating through epochs for training
    for epoch in range(epochs):
        #forward
        out = model.forward(X_train)
        #loss
        if(epoch % log_freq == 0):
            #testing model on train set and storing loss value for logging purposes
            loss_train_value = loss_func(Y_train, out[-1])
            loss_train.append(loss_train_value)
            #testing model on test set and storing loss value for logging purposes
            loss_test_value = loss_func(Y_test, model.forward(X_test)[-1])
            loss_test.append(loss_test_value)
            #
            weights_list.append([w.copy() for w in model.weights_list])
            biases_list.append([b.copy() for b in model.biases_list])
        #loss der
        loss_der = loss_derivative(Y_train, out[-1])
        #backward(update)
        model.backward(A = out, loss_derivative = loss_der, reg_layer = reg_layer, learning_rate = learning_rate, lambda_ = lambda_)
    #returning output
    return loss_train, loss_test, weights_list, biases_list
#
def train_SGD(model: FFNN, X_train:np.ndarray, Y_train:np.ndarray,X_test:np.ndarray, Y_test:np.ndarray, reg_layer:list, batch_size: int = 32,loss_info: list = [mse_loss, mse_loss_derivative], lambda_: float = 0, learning_rate: float = 0.01, epochs: int = 1000, log_freq: int = 100) -> FFNN:
    #getting loss function and its derivative
    loss_func = loss_info[0]
    loss_derivative = loss_info[1]
    #declaring variable to store loss values for logging purposes
    loss_train = []
    loss_test = []
    weights_list = []
    biases_list = []
    #getting number of samples for shuffling and batching
    num_samples = X_train.shape[0]
    #iterating through epochs for training
    for epoch in range(epochs):
        #shuffling data at the beginning of each epoch
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        X_shuffled = X_train[indices]
        Y_shuffled = Y_train[indices]
        #iterating through batches
        for i in range(0, num_samples, batch_size):
            #getting current batch of data
            X_batch = X_shuffled[i:i+batch_size]
            Y_batch = Y_shuffled[i:i+batch_size]
            #forward
            out = model.forward(X_batch)
            #loss
            if(epoch % log_freq == 0 and i == 0): 
                #testing model on train set and storing loss value for logging purposes
                loss_train_value = loss_func(Y_batch, out[-1])
                loss_train.append(loss_train_value)
                #testing model on test set and storing loss value for logging purposes
                loss_test_value = loss_func(Y_test, model.forward(X_test)[-1])
                loss_test.append(loss_test_value)
                #
                weights_list.append([w.copy() for w in model.weights_list])
                biases_list.append([b.copy() for b in model.biases_list])
            #loss der
            loss_der = loss_derivative(Y_batch, out[-1])
            #backward(update)
            model.backward(A = out, loss_derivative = loss_der, reg_layer = reg_layer, learning_rate = learning_rate, lambda_ = lambda_)
    #returning output
    return loss_train, loss_test, weights_list, biases_list