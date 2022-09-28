import os
from os import path as osp

import sklearn.linear_model as sklin
import sklearn.ensemble as ske
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

########################################################################
###                              MODELS                              ###
########################################################################
# CLASSIFICATION_MODELS = ['logistic', 'poly2_logistic', 'random_forest', 'poly2_random_forest', 'ridge_cls', 'poly2_ridge_cls', 'poly1_gb_cls']
# REGRESSION_MODELS = ['linear', 'ridge', 'poly2_ridge', 'lasso', 'poly2_lasso', 'poly1_gb_reg']
CLASSIFICATION_MODELS = ['logistic', 'ridge_cls', 'extra_trees_cls', 'random_forest_cls', 'gb_cls']
REGRESSION_MODELS = ['linear', 'ridge', 'lasso', 'extra_trees_reg', 'random_forest_reg', 'gb_reg']

#######################################################################
###                           POLYNOMIALS                           ###
#######################################################################
POLY1 = ("poly1", PolynomialFeatures())
# POLY2 = ("poly2", PolynomialFeatures())
# POLY3 = ("poly3", PolynomialFeatures())



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
}

# POLY2_LOGISTIC_REGRESSION_PARAMETERS = {
#     "poly2__degree": [2],
#     "poly2__interaction_only": [False],
#     "logistic__penalty" : ["l2"],
#     "logistic__C" : [5, 3, 1, 0.5, 0.3, 0.1],
#     "logistic__class_weight": [None, "balanced"],
#     "logistic__solver" : ["newton-cg", "lbfgs"],
#     "logistic__max_iter" : [5000],
# }

# POLY3_LOGISTIC_REGRESSION_PARAMETERS = {
#     "poly3__degree": [3],
#     "poly3__interaction_only": [False],
#     "logistic__penalty" : ["l2"],
#     "logistic__C" : [500, 300, 100, 50, 30, 10, 5],
#     "logistic__class_weight": [None, "balanced"],
#     "logistic__solver" : ["lbfgs"],
#     "logistic__max_iter" : [5000],
# }
# POLY2_LOGISTIC_PIPELINE = Pipeline(steps=[POLY2, ("logistic", sklin.LogisticRegression(fit_intercept=True))])
# POLY3_LOGISTIC_PIPELINE = Pipeline(steps=[POLY3, ("logistic", sklin.LogisticRegression(fit_intercept=True))])


#######################################################################
###                              RIDGE                              ###
#######################################################################
RIDGE_CLS_PARAMETERS = {
    "alpha": [1, 0.5, 0.3, 0.1, 0.05, 0.03, 0.01, 0.005],
    "class_weight": [None, "balanced"],
    "solver": ["svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"]
}
# POLY2_RIDGE_CLASSIFICATION_PARAMETERS = {
#     "poly2__degree": [2],
#     "poly2__interaction_only": [False],
#     "ridge_cls__alpha": [50, 30, 10, 5, 3, 1, 0.5],
#     "ridge_cls__class_weight": [None, "balanced"],
#     "ridge_cls__solver": ["svd", "cholesky", "lsqr", "sparse_cg"]
# }
# POLY3_RIDGE_CLASSIFICATION_PARAMETERS = {
#     "poly3__degree": [3],
#     "poly3__interaction_only": [False],
#     "ridge_cls__alpha": [500, 400, 300, 200, 100, 75, 50],
#     "ridge_cls__class_weight": [None, "balanced"],
#     "ridge_cls__solver": ["cholesky", "lsqr"],
# }
# POLY2_RIDGE_CLS_PIPELINE = Pipeline(steps=[POLY2, ("ridge_cls", sklin.RidgeClassifier(fit_intercept=True))])
# POLY3_RIDGE_CLS_PIPELINE = Pipeline(steps=[POLY3, ("ridge_cls", sklin.RidgeClassifier(fit_intercept=True))])

#######################################################################
###                          RANDOM FOREST                          ###
#######################################################################
RANDOM_FOREST_CLS_PARAMETERS = {
    "n_estimators": [20, 25, 20, 35, 40, 45, 50, 60, 70, 80, 90, 100],
    "criterion": ["gini", "entropy"],
    "max_depth": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 25],
    "class_weight": ["balanced"],
}
# POLY2_RANDOM_FOREST_PARAMETERS = {
#     "poly2__degree": [2],
#     "poly2__interaction_only": [False],
#     "random_forest__n_estimators": [20, 25, 20, 35, 40, 45, 50, 60, 70, 80, 90, 100],
#     "random_forest__criterion": ["entropy"],
#     "random_forest__max_depth": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 25],
#     "random_forest__class_weight": ["balanced"],
# }
# POLY2_RANDOM_FOREST_PIPELINE = Pipeline(steps=[POLY2, ("random_forest", ske.RandomForestClassifier())])


#######################################################################
###                           Extra Trees                           ###
#######################################################################
EXTRA_TREES_CLS_PARAMETERS = {
    "criterion": ["gini", "entropy"],
    "n_estimators": [60, 70, 80, 90, 100, 110, 125, 150],
    "max_depth": [14, 16, 18, 20, 22, 25, 30],
    "class_weight": [None, "balanced"],
    "random_state": [0]
}

#######################################################################
###                        GRADIENT BOOSTING                        ###
#######################################################################
GB_CLS_PARAMETERS = {
    "loss": ["exponential"],
    "learning_rate": [0.15, 0.2, 0.25, 0.3, 0.35, 0.4],
    "n_estimators": [80, 100, 125, 150, 175, 200],
    "criterion": ["friedman_mse"],
    "max_depth": [7, 8, 9, 10, 11, 12, 13],
#     "gb_cls__ccp_alpha": [0.0, 0.5], #[0.0, 0.25, 0.5, 1.0],
    "random_state": [0],
}
# POLY1_GB_CLS_PIPELINE = Pipeline(steps=[POLY1, ("gb_cls", ske.GradientBoostingClassifier())])


########################################################################
###                            REGRESSION                            ###
########################################################################

########################################################################
###                              LINEAR                              ###
########################################################################
LINEAR_REG_PARAMETERS = {}


#######################################################################
###                              RIDGE                              ###
#######################################################################
RIDGE_REG_PARAMETERS = {
    "alpha": [50, 30, 10, 5, 3, 1],
    "solver": ["svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"]
}
# POLY2_RIDGE_REGRESSION_PARAMETERS = {
#     "poly2__degree": [2],
#     "poly2__interaction_only": [False],
#     "ridge__alpha": [200, 150, 100, 75, 50, 30],
#     "ridge__solver": ["svd", "cholesky", "lsqr", "sparse_cg"]
# }
# POLY3_RIDGE_REGRESSION_PARAMETERS = {
#     "poly3__degree": [3],
#     "poly3__interaction_only": [False],
#     "ridge__alpha": [20000, 15000, 10000, 7500, 5000, 3000],
#     "ridge__solver": ["cholesky", "lsqr"]
# }
# POLY2_RIDGE_PIPELINE = Pipeline(steps=[POLY2, ("ridge", sklin.Ridge(fit_intercept=True))])
# POLY3_RIDGE_PIPELINE = Pipeline(steps=[POLY3, ("ridge", sklin.Ridge(fit_intercept=True))])


#######################################################################
###                              LASSO                              ###
#######################################################################
LASSO_REG_PARAMETERS = {
    "alpha": [1e-5, 3e-5, 6e-5, 1e-4, 3e-4, 6e-4, 1e-3, 3e-3, 6e-3, 1e-2, 3e-2, 6e-2]
}
# POLY2_LASSO_REGRESSION_PARAMETERS = {
#     "poly2__degree": [2],
#     "poly2__interaction_only": [False],
#     "lasso__alpha": [1e-5, 3e-5, 6e-5, 1e-4, 3e-4, 6e-4, 1e-3, 3e-3, 6e-3, 1e-2]
# }
# POLY2_LASSO_PIPELINE = Pipeline(steps=[POLY2, ("lasso", sklin.Lasso(fit_intercept=True, max_iter=5000))])


#######################################################################
###                           Extra Trees                           ###
#######################################################################
EXTRA_TREES_REG_PARAMETERS = {
    "criterion": ["mse"],
    "n_estimators": [60, 70, 80, 90, 100, 110, 125, 150],
    "max_depth": [14, 16, 18, 20, 22, 25, 30],
    "random_state": [0]
}


#######################################################################
###                          Random Forest                          ###
#######################################################################
RANDOM_FOREST_REG_PARAMETERS = {
    "criterion": ["mse"],
    "n_estimators": [60, 70, 80, 90, 100, 110, 125, 150],
    "max_depth": [12, 14, 16, 18, 20, 22, 25],
    "random_state": [0]
}

#######################################################################
###                        GRADIENT BOOSTING                        ###
#######################################################################
GB_REG_PARAMETERS = {
    "loss": ["ls"],
    "learning_rate": [0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11],
    "n_estimators": [80, 100, 125, 150],
    "criterion": ["friedman_mse"],
    "max_depth": [6, 7, 8, 9, 10],
#     "gb_reg__ccp_alpha": [0.0, 0.5], #[0.0, 0.25, 0.5, 1.0],
    "random_state": [0],
}
# POLY1_GB_REG_PIPELINE = Pipeline(steps=[POLY1, ("gb_reg", ske.GradientBoostingRegressor())])


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
    #"gb_classifier" : gb.gb_classifier_parameter_selection,
    "logistic": {
        "type": "classification",
        "model": sklin.LogisticRegression(fit_intercept=True),
        "params": LOGISTIC_REG_PARAMETERS,
        "pipeline": None,
        "scores": CLASSIFICATION_SCORES,
        "refit": "Avg Precision", # test if F1 Macro is better
    },
#     "poly2_logistic": {
#         "type": "classification",
#         "model": sklin.LogisticRegression(fit_intercept=True),
#         "params": POLY2_LOGISTIC_REGRESSION_PARAMETERS,
#         "pipeline": POLY2_LOGISTIC_PIPELINE,
#         "scores": CLASSIFICATION_SCORES,
#         "refit": "F1 Macro"
#     },
#     "poly3_logistic": {
#         "type": "classification",
#         "model": sklin.LogisticRegression(fit_intercept=True),
#         "params": POLY3_LOGISTIC_REGRESSION_PARAMETERS,
#         "pipeline": POLY3_LOGISTIC_PIPELINE,
#         "scores": CLASSIFICATION_SCORES,
#         "refit": "F1 Macro"
#     },
    "random_forest_cls": {
        "type": "classification",
        "model": ske.RandomForestClassifier(),
        "params": RANDOM_FOREST_CLS_PARAMETERS,
        "pipeline": None,
        "scores": CLASSIFICATION_SCORES,
        "refit": "Avg Precision", # test if F1 Macro is better
    },
    "extra_trees_cls": {
        "type": "classification",
        "model": ske.ExtraTreesClassifier(),
        "params": EXTRA_TREES_CLS_PARAMETERS,
        "pipeline": None,
        "scores": CLASSIFICATION_SCORES,
        "refit": "Avg Precision", # test if F1 Macro is better
    },
#     "poly2_random_forest": {
#         "type": "classification",
#         "model": ske.RandomForestClassifier(),
#         "params": POLY2_RANDOM_FOREST_PARAMETERS,
#         "pipeline": POLY2_RANDOM_FOREST_PIPELINE,
#          "scores": CLASSIFICATION_SCORES,
#         "refit": "F1 Macro"
#     },
    "ridge_cls": {
        "type": "classification",
        "model": sklin.RidgeClassifier(fit_intercept=True),
        "params": RIDGE_CLS_PARAMETERS,
        "pipeline": None,
        "scores": CLASSIFICATION_SCORES,
        "refit": "Avg Precision",
    },
#     "poly2_ridge_cls": {
#         "type": "classification",
#         "model": sklin.RidgeClassifier(fit_intercept=True),
#         "params": POLY2_RIDGE_CLASSIFICATION_PARAMETERS,
#         "pipeline": POLY2_RIDGE_CLS_PIPELINE,
#         "scores": CLASSIFICATION_SCORES,
#         "refit": "F1 Macro"
#     },
#     "poly3_ridge_cls": {
#         "type": "classification",
#         "model": sklin.RidgeClassifier(fit_intercept=True),
#         "params": POLY3_RIDGE_CLASSIFICATION_PARAMETERS,
#         "pipeline": POLY3_RIDGE_CLS_PIPELINE,
#         "scores": CLASSIFICATION_SCORES,
#         "refit": "F1 Macro"
#     },
    "gb_cls": {
        "type": "classification",
        "model": ske.GradientBoostingClassifier(),
        "params": GB_CLS_PARAMETERS,
        "pipeline": None,
        "scores": CLASSIFICATION_SCORES,
        "refit": "Avg Precision"
    },
    "linear": {
        "type": "regression",
        "model": sklin.LinearRegression(fit_intercept=True),
        "params": LINEAR_REG_PARAMETERS,
        "pipeline": None,
        "scores": REGRESSION_SCORES,
        "refit": "R2",
    },
    "ridge": {
        "type": "regression",
        "model": sklin.Ridge(fit_intercept=True),
        "params": RIDGE_REG_PARAMETERS,
        "pipeline": None,
        "scores": REGRESSION_SCORES,
        "refit": "R2",
    },
#     "poly2_ridge": {
#         "type": "regression",
#         "model": sklin.Ridge(fit_intercept=True),
#         "params": POLY2_RIDGE_REGRESSION_PARAMETERS,
#         "pipeline": POLY2_RIDGE_PIPELINE,
#         "scores": REGRESSION_SCORES,
#         "refit": "R2"
#     },
#     "poly3_ridge": {
#         "type": "regression",
#         "model": sklin.Ridge(fit_intercept=True),
#         "params": POLY3_RIDGE_REGRESSION_PARAMETERS,
#         "pipeline": POLY3_RIDGE_PIPELINE,
#         "scores": REGRESSION_SCORES,
#         "refit": "R2"
#     },
    "lasso": {
        "type": "regression",
        "model": sklin.Lasso(fit_intercept=True),
        "params": LASSO_REG_PARAMETERS,
        "pipeline": None,
        "scores": REGRESSION_SCORES,
        "refit": "R2",
    },
    "extra_trees_reg": {
        "type": "regression",
        "model": ske.ExtraTreesRegressor(),
        "params": EXTRA_TREES_REG_PARAMETERS,
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
#     "poly2_lasso": {
#         "type": "regression",
#         "model": sklin.Lasso(fit_intercept=True),
#         "params": POLY2_LASSO_REGRESSION_PARAMETERS,
#         "pipeline": POLY2_LASSO_PIPELINE,
#         "scores": REGRESSION_SCORES,
#         "refit": "R2",
#     },
    "gb_reg": {
        "type": "regression",
        "model": ske.GradientBoostingRegressor(),
        "params": GB_REG_PARAMETERS,
        "pipeline": None,
        "scores": REGRESSION_SCORES,
        "refit": "R2",
    },
}