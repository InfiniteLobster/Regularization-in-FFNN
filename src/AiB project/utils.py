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
    #
    random_state = np.random.RandomState(seed)
    #
    num_samples = len(indices_in)
    #
    split_size = num_samples // n_splits
    #
    ind_folds = []
    #
    random_state.shuffle(indices_in)
    #
    for iSplit in range(n_splits):
        #
        if iSplit < n_splits - 1:
            #
            start = iSplit * split_size
            end = (iSplit + 1) * split_size
            #
            indices_fold = indices_in[start:end]   
        else:
            #
            start = iSplit * split_size
            end = None
            indices_fold = indices_in[start:end]
        #
        ind_folds.append(indices_fold)
    #
    return ind_folds
#
def train_test_folds(ind_folds, iTest):
    #
    ind_fold_test = ind_folds[iTest]
    #
    ind_fold_train = np.concatenate([ind_folds[i] for i in range(len(ind_folds)) if i != iTest])
    #
    return ind_fold_train, ind_fold_test
#
def init_from_str(method_init_str: str) -> callable:
    #
    match method_init_str:
        case "zero":
            init = init_zero
        case "xavier_uniform":
            init = init_xavier_uniform
        case _:
            raise ValueError(f"Unknown initialization method: {method_init_str}")
    #
    return init
#
def activ_from_str(activ_str: str) -> tuple[callable, callable]:
    #
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
    #
    return activ, activ_deriv
#
def activ_info_from_str_list(activ_info_str_list: list) -> list:
    #
    activ_list = []
    dactiv_list = []
    #
    for activ_str in activ_info_str_list:
        #
        activ, activ_deriv = activ_from_str(activ_str)
        #
        activ_list.append(activ)
        dactiv_list.append(activ_deriv)
    #
    return [activ_list, dactiv_list]
#
def loss_info_from_str(loss_info_str: str) -> list:
    #
    match loss_info_str:
        case "mse_loss":
            loss_func = mse_loss
            loss_derivative = mse_loss_derivative
        case _:
            raise ValueError(f"Unknown loss function: {loss_info_str}")
    #
    return [loss_func, loss_derivative]