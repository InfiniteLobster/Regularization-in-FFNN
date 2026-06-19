#--------------------Libraries--------------------#
import numpy as np
#------------------Py files code------------------#
from init import init_zero
#---------------------Classes---------------------#
class FFNN:
    #self.weights_list: [np.ndarray]
    #self.biases_list: [np.ndarray]
    #self.activ_list: [callable] -list of activation functions for each layer (except output layer)
    #self.dactiv_list: [callable] -list of derivatives of activation functions for each layer 
    def __init__(self, input_size: int, output_size: int, hidden_info: list, activ_info: list, method_init: callable):
        #Declaration of lists for weights and biases of each layer
        self.weights_list = []
        self.biases_list = []
        #initialization of parameters for each layer (weights and biases) 
        for i in range(len(hidden_info) + 1):
            if i == 0:#first layer (input layer)
                in_size = input_size
                out_size = hidden_info[0]
            elif i == len(hidden_info):#last layer (output layer)
                in_size = hidden_info[-1]
                out_size = output_size
            else:#hidden layers
                in_size = hidden_info[i-1]
                out_size = hidden_info[i]
            #initialization of weights and biases for current layer
            biases = init_zero([1, out_size], dtype=np.float32)
            weights = method_init([out_size, in_size], dtype=np.float32)
            #adding weights and biases to lists
            self.weights_list.append(weights)
            self.biases_list.append(biases)
        #passing activation functions list to class variable for use in forward pass
        self.activ_list = activ_info[0]
        self.dactiv_list = activ_info[1]
    def forward(self, X: np.ndarray) -> np.ndarray:
        #
        a = X
        #forward pass through the network using weights, biases and activation functions
        for i in range(len(self.weights_list)):
            #
            weights = self.weights_list[i]
            biases = self.biases_list[i]
            activ_func = self.activ_list[i] 
            #linear transformation
            z =  a @ weights.T + biases
            #activation function (except for output layer)
            a = activ_func(z)
        return a
    def backward(self, X: np.ndarray, y: np.ndarray) -> None:
        #backward pass through the network using derivatives of activation functions and weights for calculating gradients for each layer
        raise NotImplementedError("Implement backward pass")

#--------------------Functions--------------------#
#Basic model building handler, i.e. model selection of implemented architectures and their configuration.
def build_model():
    raise NotImplementedError("Implement model creation")
