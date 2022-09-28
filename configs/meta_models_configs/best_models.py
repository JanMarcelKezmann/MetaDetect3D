import os
from os import path as osp

import sklearn.linear_model as sklin
import sklearn.ensemble as ske
import sklearn.neural_network as nn

########################################################################
###                              MODELS                              ###
########################################################################
CLASSIFICATION_MODELS = ['logistic', 'ridge_cls', 'random_forest_cls', 'gb_cls', "mlp_cls"]
REGRESSION_MODELS = ['ridge_reg', 'lasso_reg', 'random_forest_reg', 'gb_reg', "mlp_reg"]

########################################################################
###                          CLASSIFICATION                          ###
########################################################################


########################################################################
###                             LOGISTIC                             ###
########################################################################
LOGISTIC_REG_PARAMETERS = {
    "penalty" : ["l2"],
    "C" : [0.5, 0.3, 0.1, 0.05, 0.01, 0.005],
    "solver" : ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
    "max_iter" : [5000],
    "random_state": [0]
}


#######################################################################
###                              RIDGE                              ###
#######################################################################
RIDGE_CLS_PARAMETERS = {
    "alpha": [1, 0.5, 0.3, 0.1, 0.05, 0.03, 0.01, 0.005],
    "class_weight": [None, "balanced"],
    "solver": ["svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"],
    "random_state": [0]
}

#######################################################################
###                          RANDOM FOREST                          ###
#######################################################################
RANDOM_FOREST_CLS_PARAMETERS = {
    "n_estimators": [60, 70, 80, 90, 100, 125],
    "criterion": ["gini", "entropy"],
    "max_depth": [16, 20, 24, 28, 32],
    "class_weight": ["balanced"],
    "random_state": [0]
}

#######################################################################
###                        GRADIENT BOOSTING                        ###
#######################################################################
GB_CLS_PARAMETERS = {
    "loss": ["exponential"],
    "learning_rate": [0.25, 0.3, 0.35],
    "n_estimators": [80, 100, 125, 150],
    "criterion": ["friedman_mse"],
    "max_depth": [7, 8, 9, 10],
    "subsample": [0.8],
    "random_state": [0],
}

########################################################################
###                      MULTI LAYER PERCEPTRON                      ###
########################################################################
MLP_CLS_PARAMETERS = {
    "hidden_layer_sizes": [[25], [50], [75], [100], [150], [200], [25, 25], [50, 25], [50, 50], [75, 25], [75, 50], [75, 75], [100, 50], [100, 25], [100, 50], [100, 75], [100, 100], [200, 25], [200, 50], [200, 100]],
    "early_stopping": [True],
    "validation_fraction": [0.2],
    "n_iter_no_change": [10, 25, 50],
    "max_iter": [500],
    "random_state": [0]
}

########################################################################
###                            REGRESSION                            ###
########################################################################


#######################################################################
###                              RIDGE                              ###
#######################################################################
RIDGE_REG_PARAMETERS = {
    "alpha": [50, 30, 10, 5, 3, 1],
    "solver": ["svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"],
    "random_state": [0]
}
#######################################################################
###                              LASSO                              ###
#######################################################################
LASSO_REG_PARAMETERS = {
    "alpha": [1e-5, 3e-5, 6e-5, 1e-4, 3e-4, 6e-4, 1e-3, 3e-3, 6e-3, 1e-2, 3e-2, 6e-2],
    "random_state": [0]
}

#######################################################################
###                          Random Forest                          ###
#######################################################################
RANDOM_FOREST_REG_PARAMETERS = {
    "criterion": ["mse"],
    "n_estimators": [80, 90, 100, 110, 125, 150],
    "max_depth": [16, 18, 20, 22, 25],
    "random_state": [0]
}

#######################################################################
###                        GRADIENT BOOSTING                        ###
#######################################################################
GB_REG_PARAMETERS = {
    "loss": ["ls"],
    "learning_rate": [0.05, 0.06, 0.08, 0.1],
    "n_estimators": [60, 80, 100],
    "criterion": ["friedman_mse"],
    "max_depth": [6, 7, 8, 9, 10],
    "subsample": [0.8],
    "random_state": [0],
}

########################################################################
###                      MULTI LAYER PERCEPTRON                      ###
########################################################################
MLP_REG_PARAMETERS = {
    "hidden_layer_sizes": [[25], [50], [75], [100], [150], [200], [25, 25], [50, 25], [50, 50], [75, 25], [75, 50], [75, 75], [100, 50], [100, 25], [100, 50], [100, 75], [100, 100], [200, 25], [200, 50], [200, 100]],
    "early_stopping": [True],
    "validation_fraction": [0.2],
    "n_iter_no_change": [10, 25, 50],
    "max_iter": [500],
    "random_state": [0]
}

########################################################################
###                              SCORES                              ###
########################################################################
CLASSIFICATION_SCORES = {
    "ACC":"accuracy",
    "Avg Precision": "average_precision",
    "F1": "f1",
    "F1 Macro": "f1_macro",
    "ROC AUC": "roc_auc"
}

REGRESSION_SCORES = {
    "MSE": "neg_mean_squared_error",
    "MAE": "neg_mean_absolute_error",
    "R2": "r2"
}


#########################################################################
###                              CONFIGS                              ###
#########################################################################
MODELS_CONFIG = {
    "logistic": {
        "type": "classification",
        "model": sklin.LogisticRegression(fit_intercept=True),
        "params": LOGISTIC_REG_PARAMETERS,
        "pipeline": None,
        "scores": CLASSIFICATION_SCORES,
        "refit": "Avg Precision",
    },
    "ridge_cls": {
        "type": "classification",
        "model": sklin.RidgeClassifier(fit_intercept=True),
        "params": RIDGE_CLS_PARAMETERS,
        "pipeline": None,
        "scores": CLASSIFICATION_SCORES,
        "refit": "Avg Precision",
    },
    "random_forest_cls": {
        "type": "classification",
        "model": ske.RandomForestClassifier(),
        "params": RANDOM_FOREST_CLS_PARAMETERS,
        "pipeline": None,
        "scores": CLASSIFICATION_SCORES,
        "refit": "Avg Precision",
    },
    "gb_cls": {
        "type": "classification",
        "model": ske.GradientBoostingClassifier(),
        "params": GB_CLS_PARAMETERS,
        "pipeline": None,
        "scores": CLASSIFICATION_SCORES,
        "refit": "Avg Precision"
    },
    "mlp_cls": {
        "type": "classification",
        "model": nn.MLPClassifier(),
        "params": MLP_CLS_PARAMETERS,
        "pipeline": None,
        "scores": CLASSIFICATION_SCORES,
        "refit": "Avg Precision"
    },
    "ridge_reg": {
        "type": "regression",
        "model": sklin.Ridge(fit_intercept=True),
        "params": RIDGE_REG_PARAMETERS,
        "pipeline": None,
        "scores": REGRESSION_SCORES,
        "refit": "R2",
    },
    "lasso_reg": {
        "type": "regression",
        "model": sklin.Lasso(fit_intercept=True),
        "params": LASSO_REG_PARAMETERS,
        "pipeline": None,
        "scores": REGRESSION_SCORES,
        "refit": "R2",
    },
    "random_forest_reg": {
        "type": "regression",
        "model": ske.RandomForestRegressor(),
        "params": RANDOM_FOREST_REG_PARAMETERS,
        "pipeline": None,
        "scores": REGRESSION_SCORES,
        "refit": "R2",
    },
    "gb_reg": {
        "type": "regression",
        "model": ske.GradientBoostingRegressor(),
        "params": GB_REG_PARAMETERS,
        "pipeline": None,
        "scores": REGRESSION_SCORES,
        "refit": "R2",
    },
    "mlp_reg": {
        "type": "regression",
        "model": nn.MLPRegressor(),
        "params": MLP_REG_PARAMETERS,
        "pipeline": None,
        "scores": REGRESSION_SCORES,
        "refit": "R2",
    },
}