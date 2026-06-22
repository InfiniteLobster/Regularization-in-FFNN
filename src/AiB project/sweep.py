#--------------------Libraries--------------------#
#hydra
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
#wandb
import wandb
#data handling
import numpy as np
#--------------------Functions--------------------#
#function to start wandb run for specific lambda value in inner loop of cross-validation for hyperparameter tuning of lambda (to log training curves information)
def start_wandb_run_lamb(cfg: DictConfig, outer_fold:int, inner_fold: int, lamb: float, input_size: int, output_size: int) -> wandb.sdk.wandb_run.Run:
    #starting wandb run for current configuration
    run = wandb.init(
        #setting up wandb info for run identification and organization in the dashboard
        project=cfg.logger.project,
        entity=cfg.logger.entity,
        group="Lambda results",
        mode=cfg.logger.get("mode", "online"),
        #config part of wandb info
        config={
            "outer_fold": outer_fold,#identification
            "inner_fold": inner_fold,#identification
            "lambda": lamb,#identification and train parameter
            "epochs": int(cfg.model.epochs),#train parameter
            "learning_rate": float(cfg.model.learning_rate),#train parameter
            "loss_func": cfg.model.train_method.loss,#train parameter
            "reg_layer": cfg.model.train_method.reg_layer,#train parameter
            "train_method": cfg.model.train_method.type,#train parameter
            "batch_size": int(cfg.model.train_method.batch_size),#train parameter
            "input_size": input_size,#model architecture parameter
            "hidden_info": cfg.model.model_info.hidden_info,#model architecture parameter
            "output_size": output_size,#model architecture parameter
            "activ_info": cfg.model.model_info.activ_info,#model architecture parameter
            "method_init": cfg.model.model_info.method_init,#model architecture parameter
        },
        settings=wandb.Settings(
            init_timeout=cfg.logger.get("init_timeout", 180),
            x_disable_stats=True,
            x_disable_meta=True,
            x_disable_machine_info=True,
        ),
        #for multiple runs in one python process as is the case in sweeping, to make sure each run is treated as a separate run in wandb and not as a continuation of previous run
        reinit=True,
    )
    #returning wandb run object
    return run
#function for starting wandb run for all inner folds of specific outer fold
def start_wandb_run_in_cv(cfg: DictConfig, outer_fold:int, input_size: int, output_size: int):
    #starting wandb run for current configuration
    run = wandb.init(
        #setting up wandb info for run identification and organization in the dashboard
        project=cfg.logger.project,
        entity=cfg.logger.entity,
        group="Inner fold results",
        mode=cfg.logger.get("mode", "online"),
        #config part of wandb info
        config={
            "outer_fold": outer_fold,#identification
            "epochs": int(cfg.model.epochs),#train parameter
            "learning_rate": float(cfg.model.learning_rate),#train parameter
            "loss_func": cfg.model.train_method.loss,#train parameter
            "reg_layer": cfg.model.train_method.reg_layer,#train parameter
            "train_method": cfg.model.train_method.type,#train parameter
            "batch_size": int(cfg.model.train_method.batch_size),#train parameter
            "input_size": input_size,#model architecture parameter
            "hidden_info": cfg.model.model_info.hidden_info,#model architecture parameter
            "output_size": output_size,#model architecture parameter
            "activ_info": cfg.model.model_info.activ_info,#model architecture parameter
            "method_init": cfg.model.model_info.method_init,#model architecture parameter
        },
        settings=wandb.Settings(
            init_timeout=cfg.logger.get("init_timeout", 180),
            x_disable_stats=True,
            x_disable_meta=True,
            x_disable_machine_info=True,
        ),
        #for multiple runs in one python process as is the case in sweeping, to make sure each run is treated as a separate run in wandb and not as a continuation of previous run
        reinit=True,
    )
    #returning wandb run object
    return run
#function for starting wandb run for specific outer fold with best lambda value from inner loop of cross-validation (to log training curves information)
def start_wandb_run_out(cfg: DictConfig, outer_fold:int, lamb: float, input_size: int, output_size: int):
    #starting wandb run for current configuration
    run = wandb.init(
        #setting up wandb info for run identification and organization in the dashboard
        project=cfg.logger.project,
        entity=cfg.logger.entity,
        group="Outer fold results",
        mode=cfg.logger.get("mode", "online"),
        #config part of wandb info
        config={
            "outer_fold": outer_fold,#identification
            "lambda": lamb,#identification and train parameter
            "epochs": int(cfg.model.epochs),#train parameter
            "learning_rate": float(cfg.model.learning_rate),#train parameter
            "loss_func": cfg.model.train_method.loss,#train parameter
            "reg_layer": cfg.model.train_method.reg_layer,#train parameter
            "train_method": cfg.model.train_method.type,#train parameter
            "batch_size": int(cfg.model.train_method.batch_size),#train parameter
            "input_size": input_size,#model architecture parameter
            "hidden_info": cfg.model.model_info.hidden_info,#model architecture parameter
            "output_size": output_size,#model architecture parameter
            "activ_info": cfg.model.model_info.activ_info,#model architecture parameter
            "method_init": cfg.model.model_info.method_init,#model architecture parameter
        },
        settings=wandb.Settings(
            init_timeout=cfg.logger.get("init_timeout", 180),
            x_disable_stats=True,
            x_disable_meta=True,
            x_disable_machine_info=True,
        ),
        #for multiple runs in one python process as is the case in sweeping, to make sure each run is treated as a separate run in wandb and not as a continuation of previous run
        reinit=True,
    )
    #returning wandb run object
    return run
#function for starting wandb run for the whole cv process (to log overall results of cv and best lambda values across folds)
def start_wandb_run_out_cv(cfg: DictConfig, input_size: int, output_size: int):
    #starting wandb run for current configuration
    run = wandb.init(
        #setting up wandb info for run identification and organization in the dashboard
        project=cfg.logger.project,
        entity=cfg.logger.entity,
        group="CV fold results",
        mode=cfg.logger.get("mode", "online"),
        #config part of wandb info
        config={
            "epochs": int(cfg.model.epochs),#train parameter
            "learning_rate": float(cfg.model.learning_rate),#train parameter
            "loss_func": cfg.model.train_method.loss,#train parameter
            "reg_layer": cfg.model.train_method.reg_layer,#train parameter
            "train_method": cfg.model.train_method.type,#train parameter
            "batch_size": int(cfg.model.train_method.batch_size),#train parameter
            "input_size": input_size,#model architecture parameter
            "hidden_info": cfg.model.model_info.hidden_info,#model architecture parameter
            "output_size": output_size,#model architecture parameter
            "activ_info": cfg.model.model_info.activ_info,#model architecture parameter
            "method_init": cfg.model.model_info.method_init,#model architecture parameter
        },
        settings=wandb.Settings(
            init_timeout=cfg.logger.get("init_timeout", 180),
            x_disable_stats=True,
            x_disable_meta=True,
            x_disable_machine_info=True,
        ),
        #for multiple runs in one python process as is the case in sweeping, to make sure each run is treated as a separate run in wandb and not as a continuation of previous run
        reinit=True,
    )
    #returning wandb run object
    return run#currently not used, but leaving for structure and possible future use
#helper function for safe logging of weight and bias histograms in wandb (to avoid crashes due to extreme outliers in the data)
def safe_hist(x, max_bins: int = 64):
    #converting input to numpy array and flattening it for histogram creation, also ensuring it's of type float for compatibility with wandb histogram function
    x = np.asarray(x, dtype=float).ravel()
    #getting min and max values of the input data for histogram creation (to check for validity of the data and to avoid issues with extreme outliers that can cause crashes in wandb histogram function)
    x_min = np.nanmin(x)
    x_max = np.nanmax(x)
    #not creating histogram if data contains non-finite values or if all values are the same (to avoid sweep crashes due to extreme outliers or invalid data)
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
        return None
    #determining number of bins for histogram based on the size of the input data and a maximum limit to avoid issues with very large data (to ensure compatibility with wandb histogram function and to avoid crashes due to too many bins)
    num_bins = min(max_bins, x.size)
    #returniong histogram for logging in wandb
    return wandb.Histogram(x, num_bins=num_bins)
#helper function for logging training curves data and weight updates in wandb for each epoch during training
def log_epoch_data(loss_train: list,loss_val: list,weights_list: list,biases_list: list,prefix: str = ""):
    #iteratin through epochs
    for epoch, (tr, va, epoch_weights, epoch_biases) in enumerate(zip(loss_train, loss_val, weights_list, biases_list)):
        #getting curve data
        log_dict = {
            f"{prefix}train/loss": float(tr),
            f"{prefix}val/loss": float(va),
        }
        #iterating through network layers
        for layer_idx, (w, b) in enumerate(zip(epoch_weights, epoch_biases)):
            #saving weights histogram for current layer for logging in wandb
            h_w = safe_hist(w)
            log_dict[f"{prefix}weights/layer_{layer_idx}"] = h_w
            #saving biases histogram for current layer for logging in wandb
            if layer_idx < len(epoch_biases) - 1:
                h_b = safe_hist(b)
                #getting weight and bias data
                log_dict[f"{prefix}biases/layer_{layer_idx}"] = h_b
            else:#if there is only one output neuron, we log the bias value as a scalar instead of histogram to avoid issues with wandb histogram function with very small data (it is not robust solution as one neuron layers would cause problems as well, but it is enough for hte use case of the project)
                log_dict[f"{prefix}biases/layer_output"] = float(b.ravel()[0])
        #logging data for current epoch in wandb
        wandb.log(log_dict, step=epoch)
#main 
@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    #importing code form modules (seprate files for better orgnization of logic)
    from data import load_data
    from train import train_GD, train_SGD
    from model import FFNN
    from utils import KSplit, train_test_folds, init_from_str, activ_info_from_str_list, loss_info_from_str
    from loss import mse_loss
    #------------------main code------------------#
    #getting model configuration
    hidden_info = cfg.model.model_info.hidden_info
    activ_info_str = cfg.model.model_info.activ_info
    method_init_str = cfg.model.model_info.method_init
    #getting training configuration
    train_method_type = cfg.model.train_method.type
    batch_size = cfg.model.train_method.batch_size
    loss_info_str = cfg.model.train_method.loss
    reg_layer_str = str(cfg.model.train_method.reg_layer)  # e.g. "1010"
    lambdas = cfg.model.lambdas
    epochs = cfg.model.epochs
    learning_rate = cfg.model.learning_rate
    #getting sweeping configuration
    log_freq = cfg.search.log_freq
    seed_split = cfg.search.seed_split
    fold_out = cfg.search.fold_out
    fold_in = cfg.search.fold_in
    #convertig strings in config to actual functions for use in model building and training
    activ_info = activ_info_from_str_list(activ_info_str)#getting activation functions and their derivatives for current configuration based on strings in config
    method_init = init_from_str(method_init_str)#getting initialization method for current configuration based on string in config
    loss_info = loss_info_from_str(loss_info_str)#getting loss function and its derivative for current configuration based on string in config
    #converting regularization layer information from number to list of integers for use in training loop to determine which layers are regularized and which are not (forced due to the YAML file naming)
    reg_layer = [int(c) for c in reg_layer_str]
    #if no regularization, then there is no need to hyperparameter tune lambda, so we set it to 0 for all runs in this case, to keep the same structure of the code and avoid errors in training loop where lambda is used
    if sum(reg_layer) == 0:
        lambdas = [0]
    #data loading
    X,Y = load_data()
    input_size = X.shape[1]#number of features in the data, needed for model building
    output_size = Y.shape[1]#number of classes in the data, needed for model building
    #getting list of indices for all samples in the data, needed for splitting into folds for cross-validation
    indices = list(range(X.shape[0]))#list of indices for all samples in the data, needed for splitting into folds for cross-validation
    folds_out = KSplit(indices, n_splits=fold_out, seed=seed_split)
    #declaring variable to store cv results
    best_lambdas_results = {}
    #outer loop of cv
    for iOutFold in range(fold_out):
        #splitting data into train and test sets based on current fold for cross-validation
        indices_fold_out_train, indices_out_fold_test = train_test_folds(folds_out, iOutFold)
        #getting inner folds for cross-validation on the train set of current outer fold
        folds_in = KSplit(indices_fold_out_train, n_splits=fold_in, seed=seed_split)
        #initializing variable to store validation scores for different lambda values for current fold in cross-validation, to be used for hyperparameter tuning of lambda based on validation performance
        val_scores = np.zeros((len(lambdas), fold_in))
        #inner loop for hyperparameter tuning of regularization strength lambda for current fold in cross-validation
        for iLambda in range(len(lambdas)):
            #getting current lambda value for regularization strength for current run in sweep
            lamb = lambdas[iLambda]
            #inner loop for cv
            for jInFold in range(fold_in):
                #getting train and validation sets for current fold in inner loop
                indices_fold_in_train, indices_fold_in_val = train_test_folds(folds_in, jInFold)
                #getting data for current fold in inner loop for training and validation based on split indices
                X_train_in = X[indices_fold_in_train]
                Y_train_in = Y[indices_fold_in_train]
                X_val_in = X[indices_fold_in_val]
                Y_val_in = Y[indices_fold_in_val]
                #declaring model
                model = FFNN(input_size = input_size, output_size = output_size, hidden_info = hidden_info, activ_info = activ_info, method_init = method_init)#model building based on current configuration (architecture and hyperparameters) for current run in sweep
                #training model based on given method
                match train_method_type:
                    case "GD":
                        loss_train, loss_test, weights_list, biases_list = train_GD(model, X_train_in, Y_train_in, X_val_in, Y_val_in, reg_layer = reg_layer, loss_info = loss_info,lambda_= lamb, learning_rate = learning_rate, epochs = epochs, log_freq = log_freq)
                    case "SGD":
                        loss_train, loss_test, weights_list, biases_list = train_SGD(model, X_train_in, Y_train_in, X_val_in, Y_val_in, reg_layer = reg_layer, batch_size = batch_size, loss_info = loss_info,lambda_= lamb, learning_rate = learning_rate, epochs = epochs, log_freq = log_freq)
                    case _:
                        raise ValueError(f"Unknown training method: {train_method_type}")
                #getting and saving validation score for current lambda and inner fold
                val_score = loss_test[-1]
                val_scores[iLambda, jInFold] = val_score
                #starting wandb run for current lambda and inner fold to log
                run = start_wandb_run_lamb(cfg, iOutFold, jInFold, lamb, input_size, output_size)
                #logging data
                log_epoch_data(loss_train, loss_test, weights_list, biases_list)
                #finishing wandb run for current lambda and inner fold
                run.finish()
        #getting best lambda value from inner loop
        val_scores_mean = np.mean(val_scores, axis=1)
        best_lambda_index = int(np.argmin(val_scores_mean))
        best_lambda = lambdas[best_lambda_index]
        #starting wandb run for current outer fold to log overall results of inner loop of cv for different lambda values
        run = start_wandb_run_in_cv(cfg, iOutFold, input_size, output_size)
        #logging validation scores for different lambda values for current fold in cross-validation in wandb
        run.summary["val_scores_mean"] = val_scores_mean.tolist()
        #finishing wandb run for current outer fold for inner loop results
        run.finish()
        #splitting data into train and test sets based on current fold for cross-validation
        X_train_out = X[indices_fold_out_train]
        Y_train_out = Y[indices_fold_out_train]
        X_test_out = X[indices_out_fold_test]
        Y_test_out = Y[indices_out_fold_test]
        #declaring model for current outer fold
        model = FFNN(input_size = input_size, output_size = output_size, hidden_info = hidden_info, activ_info = activ_info, method_init = method_init)#model building based on current configuration (architecture and hyperparameters) for current run in sweep
        #training model based
        match train_method_type:
            case "GD":
                loss_train, loss_test, weights_list, biases_list = train_GD(model, X_train_out, Y_train_out, X_test_out, Y_test_out, reg_layer = reg_layer, loss_info = loss_info,lambda_= best_lambda, learning_rate = learning_rate, epochs = epochs, log_freq = log_freq)
            case "SGD":
                loss_train, loss_test, weights_list, biases_list = train_SGD(model, X_train_out, Y_train_out, X_test_out, Y_test_out, reg_layer = reg_layer, batch_size = batch_size, loss_info = loss_info,lambda_= best_lambda, learning_rate = learning_rate, epochs = epochs, log_freq = log_freq)
            case _:
                raise ValueError(f"Unknown training method: {train_method_type}")
        #saving test score for best lambda value for current fold
        best_lambdas_results[best_lambda] = loss_test[-1]
        #starting wandb run for current outer fold to log
        run =start_wandb_run_out(cfg, iOutFold, best_lambda, input_size, output_size)
        #logging data
        log_epoch_data(loss_train, loss_test, weights_list, biases_list)
        #finishing wandb run for current outer fold
        run.finish()
    #getting best lambda information for cv run
    best_lambda_overall_index = int(np.argmin(list(best_lambdas_results.values())))
    best_lambda_overall = list(best_lambdas_results.keys())[best_lambda_overall_index]
    best_lambda_overall_score = list(best_lambdas_results.values())[best_lambda_overall_index]
    #getting worst lambda information for cv run
    worst_lambda_overall_index = int(np.argmax(list(best_lambdas_results.values())))
    worst_lambda_overall = list(best_lambdas_results.keys())[worst_lambda_overall_index]
    worst_lambda_overall_score = list(best_lambdas_results.values())[worst_lambda_overall_index]
    #getting which lambda value is the most common among the best lambda values
    most_common_lambda = max(set(list(best_lambdas_results.keys())), key=list(best_lambdas_results.keys()).count)
    most_common_lambda_score = best_lambdas_results[most_common_lambda]
    #getting average test score across folds for best lambda values
    average_test_score = np.mean(list(best_lambdas_results.values()))
    #starting wandb run for whole cv process to log overall results of cv and best lambda values across folds
    run = start_wandb_run_out_cv(cfg, input_size, output_size)
    #logging overall results of cv and best lambda values across folds in wandb
    wandb.log({
        "best_lambda_overall": best_lambda_overall, 
        "best_lambda_overall_score": best_lambda_overall_score,
        "worst_lambda_overall": worst_lambda_overall,
        "worst_lambda_overall_score": worst_lambda_overall_score,
        "most_common_lambda": most_common_lambda,
        "most_common_lambda_score": most_common_lambda_score,
        "average_test_score": average_test_score
        })
    #finishing wandb run for whole cv process
    run.finish()

if __name__ == "__main__":
    main()

