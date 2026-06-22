#--------------------Libraries--------------------#
import numpy as np
#------------------Py files code------------------#
from init import init_zero, init_xavier_uniform
from activ import relu, sigmoid, relu_derivative, sigmoid_derivative, linear, linear_derivative
from loss import mse_loss, mse_loss_derivative
from update import standard, reg_L1, reg_L2
#--------------------Functions--------------------#
#Function to get the project root directory
def KSplit(indices_in: list, n_splits:int=5, seed:int=42): 
    #getting random state for reproducibility of splits
    random_state = np.random.RandomState(seed)
    #getting the number of samples in the input indices list
    num_samples = len(indices_in)
    #getting the size of each split based on the number of samples and the number of splits
    split_size = num_samples // n_splits
    #declaring variable to store the indices for each fold
    ind_folds = []
    #shuffling the input indices list to ensure random distribution of samples across folds
    random_state.shuffle(indices_in)
    #iterating through the number of splits to create the folds
    for iSplit in range(n_splits):
        #determining the start and end indices for the current fold
        if iSplit < n_splits - 1:#each split except the last one will have the same number of samples, the last split will take the remaining samples to ensure all samples are included in the folds
            #getting the start and end indices for the current fold
            start = iSplit * split_size
            end = (iSplit + 1) * split_size
            #
            indices_fold = indices_in[start:end]   
        else:
            #getting the start index for the last fold based 
            start = iSplit * split_size
            end = None#for the last fold, the end index is set to None to include all remaining samples in the fold
            indices_fold = indices_in[start:end]
        #appending the indices for the current fold to the list of folds
        ind_folds.append(indices_fold)
    #returning output
    return ind_folds
#function to get the train and test indices for a given fold in cross-validation
def train_test_folds(ind_folds, iTest):
    #getting the test indices for the current fold based on the input fold index
    ind_fold_test = ind_folds[iTest]
    #getting all other fold indices except the current fold index to be used as train indices for the current fold
    ind_fold_train = np.concatenate([ind_folds[i] for i in range(len(ind_folds)) if i != iTest])
    #returning output
    return ind_fold_train, ind_fold_test
#function to convert string input from the confing into callable function used by the code
def init_from_str(method_init_str: str) -> callable:
    #getting the initialization function based on the input string for initialization method
    match method_init_str:
        case "zero":
            init = init_zero
        case "xavier_uniform":
            init = init_xavier_uniform
        case _:
            raise ValueError(f"Unknown initialization method: {method_init_str}")
    #returning output
    return init
#function to convert string input from the confing into callable function used by the code for activation function and its derivative
def activ_from_str(activ_str: str) -> tuple[callable, callable]:
    #getting the activation function and its derivative based on the input string for activation function
    match activ_str:
        case "relu":
            activ = relu
            activ_deriv = relu_derivative
        case "sigmoid":
            activ = sigmoid
            activ_deriv = sigmoid_derivative
        case "linear":
            activ = linear  
            activ_deriv = linear_derivative
        case _:
            raise ValueError(f"Unknown activation function: {activ_str}")
    #returning output
    return activ, activ_deriv
#function to convert string input from the confing into list of callable functions used by the code for list of activation functions and their derivatives for different layers in the network
def activ_info_from_str_list(activ_info_str_list: list) -> list:
    #declaring variables to store the list of activation functions and their derivatives for different layers in the network
    activ_list = []
    dactiv_list = []
    #iterating through input of layers
    for activ_str in activ_info_str_list:
        #getting the activation function and its derivative for the current layer
        activ, activ_deriv = activ_from_str(activ_str)
        #appending the activation function and its derivative to the respective lists
        activ_list.append(activ)
        dactiv_list.append(activ_deriv)
    #returning output
    return [activ_list, dactiv_list]
#function to convert string input from the confing into callable function used by the code for loss function and its derivative
def loss_info_from_str(loss_info_str: str) -> list:
    #getting the loss function and its derivative based on the input string for loss function
    match loss_info_str:
        case "mse_loss":
            loss_func = mse_loss
            loss_derivative = mse_loss_derivative
        case _:
            raise ValueError(f"Unknown loss function: {loss_info_str}")
    #
    return [loss_func, loss_derivative]