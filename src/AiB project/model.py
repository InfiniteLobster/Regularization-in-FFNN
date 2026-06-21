#--------------------Libraries--------------------#
import numpy as np
#------------------Py files code------------------#
from init import init_zero
from update import standard
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
        #creating list for storing outputs of each layer during forward pass (including input)
        A = []
        #passing input to the mult variable for forward pass
        A.append(X)
        #forward pass through the network using weights, biases and activation functions
        for i in range(len(self.weights_list)):
            #getting weights, biases and activation function for current layer from the instance variables
            weights = self.weights_list[i]
            biases = self.biases_list[i]
            activ_func = self.activ_list[i] 
            #linear transformation
            z =  A[-1] @ weights.T + biases
            #activation function (except for output layer)
            a = activ_func(z)
            #adding output of current layer to the list for use in backward pass
            A.append(a)
        #returning output
        return A
    def backward(self, A: list, loss_derivative: np.ndarray, update_func: callable, learning_rate: float, lambda_: float = 0) -> None:
        #calculating delta for output layer
        delta = loss_derivative * self.dactiv_list[-1](A[-1])
        #iterating through layers in reverse order for backpropagation
        for i in reversed(range(len(self.weights_list))):
            #getting weights and activation function derivative for current layer from the instance variables
            weights = self.weights_list[i]
            dactiv_func = self.dactiv_list[i]
            #getting activation output for current layer from the forward pass results
            a = A[i]
            #calculating gradients for weights and biases
            grad_weights = delta.T @ a
            grad_biases = np.sum(delta, axis=0, keepdims=True)
            #updating weights and biases using gradients (here we can add learning rate and regularization)
            self.weights_list[i] -= update_func(grad_weights, learning_rate, self.weights_list[i], lambda_)
            self.biases_list[i] -= standard(grad_biases, learning_rate, self.biases_list[i], lambda_) #biases are not regularized, only weights
            #calculating delta for previous layer (except for input layer)
            if i > 0:
                next_delta = None
                next_delta = (delta @ weights) * dactiv_func(a)
                delta = next_delta
#--------------------Functions--------------------#
#Basic model building handler, i.e. model selection of implemented architectures and their configuration.
def build_model():
    raise NotImplementedError("Implement model creation")
