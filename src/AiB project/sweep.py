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
#function for starting wandb run 
def start_wandb_run_lamb(cfg: DictConfig, outer_fold:int, inner_fold: int, lamb: float):
    #starting wandb run for current configuration
    run = wandb.init(
        #setting up wandb info for run identification and organization in the dashboard
        project=cfg.logger.project,
        entity=cfg.logger.entity,
        group="Lambda results",
        mode=cfg.logger.get("mode", "online"),
        #config part of wandb info
        config={
            "outer_fold": outer_fold,
            "inner_fold": inner_fold,
            "lambda": lamb,
            "epochs": int(cfg.model.epochs),
            "learning_rate": float(cfg.model.learning_rate),
            "reg_mode": cfg.model.train_method.reg_mode,
            "train_method": cfg.model.train_method.type,
            "batch_size": int(cfg.model.train_method.batch_size),
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
#function for starting wandb run 
def start_wandb_run_in_cv(cfg: DictConfig, outer_fold:int):
    #starting wandb run for current configuration
    run = wandb.init(
        #setting up wandb info for run identification and organization in the dashboard
        project=cfg.logger.project,
        entity=cfg.logger.entity,
        group="Inner fold results",
        mode=cfg.logger.get("mode", "online"),
        #config part of wandb info
        config={
            "outer_fold": outer_fold,
            "epochs": int(cfg.model.epochs),
            "learning_rate": float(cfg.model.learning_rate),
            "reg_mode": cfg.model.train_method.reg_mode,
            "train_method": cfg.model.train_method.type,
            "batch_size": int(cfg.model.train_method.batch_size),
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
def start_wandb_run_out(cfg: DictConfig, outer_fold:int, lamb: float):
    #starting wandb run for current configuration
    run = wandb.init(
        #setting up wandb info for run identification and organization in the dashboard
        project=cfg.logger.project,
        entity=cfg.logger.entity,
        group="Outer fold results",
        mode=cfg.logger.get("mode", "online"),
        #config part of wandb info
        config={
            "outer_fold": outer_fold,
            "lambda": lamb,
            "epochs": int(cfg.model.epochs),
            "learning_rate": float(cfg.model.learning_rate),
            "reg_mode": cfg.model.train_method.reg_mode,
            "train_method": cfg.model.train_method.type,
            "batch_size": int(cfg.model.train_method.batch_size),
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
def start_wandb_run_out_cv(cfg: DictConfig):
    #starting wandb run for current configuration
    run = wandb.init(
        #setting up wandb info for run identification and organization in the dashboard
        project=cfg.logger.project,
        entity=cfg.logger.entity,
        group="CV fold results",
        mode=cfg.logger.get("mode", "online"),
        #config part of wandb info
        config={
            "epochs": int(cfg.model.epochs),
            "learning_rate": float(cfg.model.learning_rate),
            "reg_mode": cfg.model.train_method.reg_mode,
            "train_method": cfg.model.train_method.type,
            "batch_size": int(cfg.model.train_method.batch_size),
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
def log_epoch_curves(loss_train:list, loss_val:list, prefix: str = "", start_epoch: int = 1) -> None:
    #
    train_losses = loss_train
    val_losses   = loss_val
    #
    for tr, va in zip(train_losses, val_losses):
        wandb.log(
            {
                f"{prefix}train/loss": float(tr),
                f"{prefix}val/loss": float(va),
            }
        )
#main 
@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    #importing code form modules (seprate files for better orgnization of logic)
    from data import load_data
    from train import train_GD, train_SGD
    from model import FFNN
    from utils import KSplit, train_test_folds, init_from_str, activ_info_from_str_list, loss_info_from_str, update_from_str
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
    reg_mode = cfg.model.train_method.reg_mode
    lambdas = cfg.model.lambdas
    epochs = cfg.model.epochs
    learning_rate = cfg.model.learning_rate
    #
    log_freq = cfg.search.log_freq
    seed_split = cfg.search.seed_split
    fold_out = cfg.search.fold_out
    fold_in = cfg.search.fold_in
    #convertig strings in config to actual functions for use in model building and training
    activ_info = activ_info_from_str_list(activ_info_str)#getting activation functions and their derivatives for current configuration based on strings in config
    method_init = init_from_str(method_init_str)#getting initialization method for current configuration based on string in config
    loss_info = loss_info_from_str(loss_info_str)#getting loss function and its derivative for current configuration based on string in config
    update_func = update_from_str(reg_mode)#getting update function for current configuration based on regularization mode string in config
    #if no regularization, then there is no need to hyperparameter tune lambda, so we set it to 0 for all runs in this case, to keep the same structure of the code and avoid errors in training loop where lambda is used
    if reg_mode == "None":
        lambdas = [0]
    #data loading
    X,Y = load_data()
    input_size = X.shape[1]#number of features in the data, needed for model building
    output_size = Y.shape[1]#number of classes in the data, needed for model building
    #getting list of indices for all samples in the data, needed for splitting into folds for cross-validation
    indices = list(range(X.shape[0]))#list of indices for all samples in the data, needed for splitting into folds for cross-validation
    folds_out = KSplit(indices, n_splits=fold_out, seed=seed_split)
    #
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
                #
                run = start_wandb_run_lamb(cfg, iOutFold, jInFold, lamb)
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
                        loss_train, loss_test = train_GD(model, X_train_in, Y_train_in, X_val_in, Y_val_in, loss_info = loss_info, update_func = update_func,lambda_= lamb, learning_rate = learning_rate, epochs = epochs, log_freq = log_freq)
                    case "SGD":
                        loss_train, loss_test = train_SGD(model, X_train_in, Y_train_in, X_val_in, Y_val_in, batch_size = batch_size, loss_info = loss_info, update_func = update_func,lambda_= lamb, learning_rate = learning_rate, epochs = epochs, log_freq = log_freq)
                    case _:
                        raise ValueError(f"Unknown training method: {train_method_type}")
                #getting and saving validation score for current lambda and inner fold
                val_score = loss_test[-1]
                val_scores[iLambda, jInFold] = val_score
                #
                log_epoch_curves(loss_train, loss_test)
                #
                run.finish()
        #getting best lambda value from inner loop
        val_scores_mean = np.mean(val_scores, axis=1)
        best_lambda_index = int(np.argmin(val_scores_mean))
        best_lambda = lambdas[best_lambda_index]
        #
        run = start_wandb_run_in_cv(cfg, iOutFold)
        #
        wandb.log({
            "best_lambda": best_lambda, 
            "best_lambda_score": val_scores_mean[best_lambda_index],
            "lambda_0_score": val_scores_mean[0], #score for lambda = 0, i.e. no regularization, for comparison
            })
        #
        run.summary["val_scores_mean"] = val_scores_mean.tolist()
        #
        run.finish()
        #splitting data into train and test sets based on current fold for cross-validation
        X_train_out = X[indices_fold_out_train]
        Y_train_out = Y[indices_fold_out_train]
        X_test_out = X[indices_out_fold_test]
        Y_test_out = Y[indices_out_fold_test]
        #
        run =start_wandb_run_out(cfg, iOutFold, best_lambda)
        #
        model = FFNN(input_size = input_size, output_size = output_size, hidden_info = hidden_info, activ_info = activ_info, method_init = method_init)#model building based on current configuration (architecture and hyperparameters) for current run in sweep
        #
        match train_method_type:
            case "GD":
                loss_train, loss_test = train_GD(model, X_train_out, Y_train_out, X_test_out, Y_test_out, loss_info = loss_info, update_func = update_func,lambda_= best_lambda, learning_rate = learning_rate, epochs = epochs, log_freq = log_freq)
            case "SGD":
                loss_train, loss_test = train_SGD(model, X_train_out, Y_train_out, X_test_out, Y_test_out, batch_size = batch_size, loss_info = loss_info, update_func = update_func,lambda_= best_lambda, learning_rate = learning_rate, epochs = epochs, log_freq = log_freq)
            case _:
                raise ValueError(f"Unknown training method: {train_method_type}")
        #
        best_lambdas_results[best_lambda] = loss_test[-1]
        #
        log_epoch_curves(loss_train, loss_test)
        #
        run.finish()
    #
    run = start_wandb_run_out_cv(cfg)
    #
    best_lambda_overall_index = int(np.argmin(list(best_lambdas_results.values())))
    best_lambda_overall = list(best_lambdas_results.keys())[best_lambda_overall_index]
    best_lambda_overall_score = list(best_lambdas_results.values())[best_lambda_overall_index]
    #
    worst_lambda_overall_index = int(np.argmax(list(best_lambdas_results.values())))
    worst_lambda_overall = list(best_lambdas_results.keys())[worst_lambda_overall_index]
    worst_lambda_overall_score = list(best_lambdas_results.values())[worst_lambda_overall_index]
    #
    average_test_score = np.mean(list(best_lambdas_results.values()))
    #
    wandb.log({
        "best_lambda_overall": best_lambda_overall, 
        "best_lambda_overall_score": best_lambda_overall_score,
        "worst_lambda_overall": worst_lambda_overall,
        "worst_lambda_overall_score": worst_lambda_overall_score,
        "average_test_score": average_test_score
        })
    #
    run.finish()

if __name__ == "__main__":
    main()

