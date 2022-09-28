import os
import shutil
import joblib
import numpy as np
import pandas as pd

from copy import deepcopy
from os import path as osp
from joblib import Parallel, delayed
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score

from . import TrainingMetrics

class Trainer(TrainingMetrics):
    def __init__(self, path_cfg = None, data_cfg = None, models_cfg = None, save_dir = None, cv: int = 5, n_jobs: int = 30, verbose: int = 5):
        super(Trainer, self).__init__(path_cfg=path_cfg, data_cfg=data_cfg, models_cfg=models_cfg)
        
        self.X_train = None
        self.X_val = None
        self.y_train_reg = None
        self.y_val_reg = None
        self.y_train_cls = None
        self.y_val_cls = None
        
        self.save_dir: str = save_dir
        self.cv: int = cv
        self.n_jobs: int = n_jobs
        self.verbose: int = verbose
            
        self.grid_results: pd.DataFrame = pd.DataFrame({})
            
        if self.save_dir is None:
            print("'save_dir' should be given, else results will not be saved.")
        
    def get_model_input(self):
        self.X_train, self.X_val, self.y_train_reg, self.y_val_reg, self.y_train_cls, self.y_val_cls = self.get_scaled_train_val_split()
        
    def parameter_search(self):
        for param_type in ["baseline", "all_params"]:
            baseline = True if param_type == "baseline" else False
            for score_thresh in self.data_cfg.meta_metrics_score_thresholds:
                for iou_thresh in self.data_cfg.meta_metrics_iou_thresholds:
                    self.update_thresholds(iou_thresh=iou_thresh, score_thresh=score_thresh, baseline=baseline)
                    self.get_model_input()

    #                 for model_name, model_config in self.models_cfg.MODELS_CONFIG.items():
                    for model_name, model_config in self.models_cfg.MODELS_CONFIG.items():                    
                        model = model_config["model"]
                        params = model_config["params"]
                        pipeline = model_config["pipeline"]
                        model_type = model_config["type"]
                        scoring = model_config["scores"]
                        refit = model_config["refit"]
                        print(model_config)

                        if pipeline is None:
                            if model_type == "regression":
                                clf = GridSearchCV(model, params, scoring=scoring, refit=refit, cv=self.cv, verbose=self.verbose, n_jobs=self.n_jobs)
                                clf.fit(self.X_train, self.y_train_reg["tp_score"])
                            else:
                                clf = GridSearchCV(model, params, scoring=scoring, refit=refit, cv=self.cv, verbose=self.verbose, n_jobs=self.n_jobs)
                                clf.fit(self.X_train, self.y_train_cls["tp_label"])
                        else:
                            if model_type == "regression":
                                clf = GridSearchCV(pipeline, params, scoring=scoring, refit=refit, cv=self.cv, verbose=self.verbose, n_jobs=self.n_jobs)
                                clf.fit(self.X_train, self.y_train_reg["tp_score"])
                            else:
                                clf = GridSearchCV(pipeline, params, scoring=scoring, refit=refit, cv=self.cv, verbose=self.verbose, n_jobs=self.n_jobs)
                                clf.fit(self.X_train, self.y_train_cls["tp_label"])

                        print(clf.best_estimator_, clf.best_score_)
                        print("\n")

    #                     self.grid_results[model_name] = {
                        results = {
                            "best_estimator": clf.best_estimator_,
                            "best_params": clf.best_params_,
                            "param_grid": clf.param_grid,
                            "best_score": clf.best_score_,
                            "scoring": clf.scoring,
                            "refit": clf.refit,
                            "cv_results": clf.cv_results_,
                        }

                        if self.save_dir is not None:
                            self.save_grid_results(model_name, results, prefix=param_type)
#                 self.save_params()

    
    def parameter_search_best_k(self):
        def fit_k_features(model_config, model_name, k_features_indices):
            model = model_config["model"].__class__()
            cv_results = self.load_grid_results(model_name, "all_params")["cv_results"]
            
            if model_config["type"] == "regression":
                best_params = np.asarray(cv_results["params"])[np.argsort(cv_results["rank_test_R2"])[0]]
                model.set_params(**best_params)
                model.fit(self.X_train[:, k_features_indices], self.y_train_reg["tp_score"])
                best_score = model.score(self.X_train[:, k_features_indices], self.y_train_reg["tp_score"])
            else:
                best_params = np.asarray(cv_results["params"])[np.argsort(cv_results["rank_test_Avg Precision"])[0]]
                model.set_params(**best_params)
                model.fit(self.X_train[:, k_features_indices], self.y_train_cls["tp_label"])
                best_score = model.score(self.X_train[:, k_features_indices], self.y_train_cls["tp_label"])
                
            return model, best_score, best_params
            
        param_type = "best_k"
        
        all_features = self.data_cfg.PRED_OUTPUT_FIELDS + self.data_cfg.MD3D_METRICS
        all_features.remove("file_path")
        all_features.remove("dataset_box_id")
        
        for score_thresh in self.data_cfg.meta_metrics_score_thresholds:
            for iou_thresh in self.data_cfg.meta_metrics_iou_thresholds:
                if iou_thresh != 0.5:
                    continue
                self.update_thresholds(iou_thresh=iou_thresh, score_thresh=score_thresh)
                self.get_model_input()
                
                for model_name, model_config in self.models_cfg.MODELS_CONFIG.items():
                    if "mlp" in model_name:
                        continue
                    best_k_features = []

                    new_feature = None
                    temp_best_score = - 1000000.0
                    pre_temp_best_score = - 1000000.0
                
                    k_features = []
                    
                    for i in range(len(all_features) - 1):
                        for feature in all_features:
                            if feature in best_k_features:
                                continue
                            k_features = deepcopy(best_k_features)
                            k_features.append(feature)
                            k_features_indices = [idx for idx, feat in enumerate(self.X.columns) if feat in k_features]
                            
                            model, best_score, _ = fit_k_features(model_config, model_name, k_features_indices)

                            if best_score > temp_best_score:
                                print(feature, best_score)
                                temp_best_score = best_score
                                new_feature = feature

                        if new_feature is not None and temp_best_score > pre_temp_best_score + 5e-4:
                            best_k_features.append(new_feature)
                            pre_temp_best_score = temp_best_score
                        else:
                            break
                        
                        if len(best_k_features) >= 10:
                            break
                    
                    best_k_features_indices = [idx for idx, feat in enumerate(self.X.columns) if feat in best_k_features]
                    model, best_score, best_params = fit_k_features(model_config, model_name, best_k_features_indices)

                    results = {
                        "best_estimator": model_config["model"],
                        "best_params": best_params,
                        "best_score": best_score,
                        "scoring": model_config["scores"],
                        "refit": model_config["refit"],
                        "best_k_features": best_k_features
                    }

                    if self.save_dir is not None:
                        self.save_grid_results(model_name, results, prefix=param_type)


    def train_eval(self):
        
        def train_reg_model(model, precision=5, k_features_indices=None, seed=None):
            if seed is not None:
                model.set_params(**{"random_state": seed})
            
            if k_features_indices is not None:
                model.fit(self.X_train[:, k_features_indices], self.y_train_reg["tp_score"])
                
                temp_result = {
                    "train_r2": round(model.score(self.X_train[:, k_features_indices], self.y_train_reg["tp_score"]), precision),
                    "val_r2": round(model.score(self.X_val[:, k_features_indices], self.y_val_reg["tp_score"]), precision),

                    "y_train_pred_reg": model.predict(self.X_train[:, k_features_indices]),
                    "y_val_pred_reg": model.predict(self.X_val[:, k_features_indices]),
                }
            else:
                model.fit(self.X_train, self.y_train_reg["tp_score"])
            
                temp_result = {
                    "train_r2": round(model.score(self.X_train, self.y_train_reg["tp_score"]), precision),
                    "val_r2": round(model.score(self.X_val, self.y_val_reg["tp_score"]), precision),

                    "y_train_pred_reg": model.predict(self.X_train),
                    "y_val_pred_reg": model.predict(self.X_val),
                }
            return temp_result
        
        def train_cls_model(model, precision=5, k_features_indices=None, seed=None):
            if seed is not None:
                model.set_params(**{"random_state": seed})
                
            if k_features_indices is not None:
                model.fit(self.X_train[:, k_features_indices], self.y_train_cls["tp_label"])

                try:
                    y_train_proba = model.predict_proba(self.X_train[:, k_features_indices])
                    y_val_proba = model.predict_proba(self.X_val[:, k_features_indices])
                except:
                    y_train_proba, y_val_proba = None, None

                y_train_pred = model.predict(self.X_train[:, k_features_indices])
                y_val_pred = model.predict(self.X_val[:, k_features_indices])

            else:
                model.fit(self.X_train, self.y_train_cls["tp_label"])

                try:
                    y_train_proba = model.predict_proba(self.X_train)
                    y_val_proba = model.predict_proba(self.X_val)
                except:
                    y_train_proba, y_val_proba = None, None

                y_train_pred = model.predict(self.X_train)
                y_val_pred = model.predict(self.X_val)

            temp_result = {
                "train_avg_precision": round(average_precision_score(y_train_pred, self.y_train_cls["tp_label"]), precision),
                "val_avg_precision": round(average_precision_score(y_val_pred, self.y_val_cls["tp_label"]), precision),

                "train_f1_macro": round(f1_score(y_train_pred, self.y_train_cls["tp_label"], average="macro"), precision),
                "val_f1_macro": round(f1_score(y_val_pred, self.y_val_cls["tp_label"], average="macro"), precision),
            }

            try:
                temp_result.update({
                    "train_roc_auc": round(roc_auc_score(y_train_pred, self.y_train_cls["tp_label"]), precision),
                    "val_roc_auc": round(roc_auc_score(y_val_pred, self.y_val_cls["tp_label"]), precision),
                })
            except:
                temp_result.update({
                    "train_roc_auc": 0.0,
                    "val_roc_auc": 0.0,
                })
            
            temp_result.update({
                "y_train_pred_proba_cls": y_train_proba,
                "y_val_pred_proba_cls": y_val_proba,

                "y_train_pred_cls": y_train_pred,
                "y_val_pred_cls": y_val_pred, 
            })
            
            return temp_result
        
        def process_reg_temp_results(temp_results_ld, precision=5):
            # converts list of dicts to dict of lists
            results_dl = {key: [d[key] for d in temp_results_ld] for key in temp_results_ld[0].keys()}

            results = {
                "train_r2_best": round(np.max(results_dl["train_r2"]), precision),
                "train_r2_mean": round(np.mean(results_dl["train_r2"]), precision),
                "train_r2_std": round(np.std(results_dl["train_r2"]), precision),
                
                "val_r2_best": round(np.max(results_dl["val_r2"]), precision),
                "val_r2_mean": round(np.mean(results_dl["val_r2"]), precision),
                "val_r2_std": round(np.std(results_dl["val_r2"]), precision),
                
                "y_train_pred_reg": results_dl["y_train_pred_reg"][np.argmax(results_dl["val_r2"])],
                "y_val_pred_reg":  results_dl["y_val_pred_reg"][np.argmax(results_dl["val_r2"])],
            }
            
            return results
        
        def process_cls_temp_results(temp_results_ld, precision=5):
            # converts list of dicts to dict of lists
            results_dl = {key: [d[key] for d in temp_results_ld] for key in temp_results_ld[0].keys()}
            
            results = {
                "train_avg_precision_best": round(np.max(results_dl["train_avg_precision"]), precision),
                "train_avg_precision_mean": round(np.mean(results_dl["train_avg_precision"]), precision),
                "train_avg_precision_std": round(np.std(results_dl["train_avg_precision"]), precision),
                
                "val_avg_precision_best": round(np.max(results_dl["val_avg_precision"]), precision),
                "val_avg_precision_mean": round(np.mean(results_dl["val_avg_precision"]), precision),
                "val_avg_precision_std": round(np.std(results_dl["val_avg_precision"]), precision),
                
                
                "train_f1_macro_best": round(np.max(results_dl["train_f1_macro"]), precision),
                "train_f1_macro_mean": round(np.mean(results_dl["train_f1_macro"]), precision),
                "train_f1_macro_std": round(np.std(results_dl["train_f1_macro"]), precision),
                
                "val_f1_macro_best": round(np.max(results_dl["val_f1_macro"]), precision),
                "val_f1_macro_mean": round(np.mean(results_dl["val_f1_macro"]), precision),
                "val_f1_macro_std": round(np.std(results_dl["val_f1_macro"]), precision),
                
                
                "train_roc_auc_best": round(np.max(results_dl["train_roc_auc"]), precision),
                "train_roc_auc_mean": round(np.mean(results_dl["train_roc_auc"]), precision),
                "train_roc_auc_std": round(np.std(results_dl["train_roc_auc"]), precision),
                
                "val_roc_auc_best": round(np.max(results_dl["val_roc_auc"]), precision),
                "val_roc_auc_mean": round(np.mean(results_dl["val_roc_auc"]), precision),
                "val_roc_auc_std": round(np.std(results_dl["val_roc_auc"]), precision),
                 
                "y_train_pred_proba_cls": results_dl["y_train_pred_proba_cls"][np.argmax(results_dl["val_avg_precision"])],
                "y_val_pred_proba_cls": results_dl["y_val_pred_proba_cls"][np.argmax(results_dl["val_avg_precision"])],

                "y_train_pred_cls": results_dl["y_train_pred_cls"][np.argmax(results_dl["val_avg_precision"])],
                "y_val_pred_cls": results_dl["y_val_pred_cls"][np.argmax(results_dl["val_avg_precision"])],
            }
            
            return results
        
        #for param_type in ["best_k"]:
        for param_type in ["baseline", "all_params"]:
            baseline = True if param_type == "baseline" else False
            for score_thresh in self.data_cfg.meta_metrics_score_thresholds:
                for iou_thresh in self.data_cfg.meta_metrics_iou_thresholds:
                    self.update_thresholds(iou_thresh=iou_thresh, score_thresh=score_thresh, baseline=baseline)
                    self.get_model_input()

                    for model_name, model_config in self.models_cfg.MODELS_CONFIG.items():
                        model = model_config["model"].__class__()

                        if param_type == "best_k":
                            if "mlp" in model_name:
                                continue
                            best_params = self.load_grid_results(model_name, param_type)["best_params"]
                            print(best_params)
                            best_k_features = self.load_grid_results(model_name, param_type)["best_k_features"]
                            print(best_k_features)
                            best_k_features_indices = [idx for idx, feat in enumerate(self.X.columns) if feat in best_k_features]
                        else:
                            cv_results = self.load_grid_results(model_name, param_type)["cv_results"]
                            if model_config["type"] == "regression":
                                best_params = np.asarray(cv_results["params"])[np.argsort(cv_results["rank_test_R2"])[0]]
                            else:
                                best_params = np.asarray(cv_results["params"])[np.argsort(cv_results["rank_test_Avg Precision"])[0]]
                            best_k_features_indices = None

    #                             for param in model.get_params().keys():
    #                                 if "n_jobs" in param:
    #                                     model.set_params(**{param: 20})
                        model.set_params(**best_params)

                        print(model)

                        if model_config["type"] == "regression":
                            if model_name in ['ridge_reg', 'lasso_reg']:
                                results = train_reg_model(model, k_features_indices=best_k_features_indices)
                            else:
                                temp_results = Parallel(n_jobs=15)(delayed(train_reg_model)(model, k_features_indices=best_k_features_indices, seed=seed) for seed in range(15))
                                results = process_reg_temp_results(temp_results)

                        else:
                            # return prediction of best model [o]
                            if model_name in ['logistic', 'ridge_cls']:
                                results = train_cls_model(model, k_features_indices=best_k_features_indices)
                            else:
                                temp_results = Parallel(n_jobs=15)(delayed(train_cls_model)(model, k_features_indices=best_k_features_indices, seed=seed) for seed in range(15))

                                results = process_cls_temp_results(temp_results)

                        if self.save_dir is not None:
                            self.save_train_eval_results(model_name, results, prefix=param_type)

    
    def save_grid_results(self, model_name, results, prefix=""):
#         for model_name, model_config in self.models_cfg.MODELS_CONFIG.items():
            # change this with subfolder getter
        param_dir = osp.join(self.save_dir, self.get_subfolder_name(iou_thresh=self.iou_thresh, score_thresh=self.score_thresh), f"{prefix}_params/")
        os.makedirs(param_dir, exist_ok=True)

        #save your model or results
        joblib.dump(results, osp.join(param_dir, f"{model_name}.pkl"))
    
    
    def load_grid_results(self, model_name, prefix=""):
        subfolder_name = self.get_subfolder_name(iou_thresh=self.iou_thresh, score_thresh=self.score_thresh)
        return joblib.load(osp.join(self.save_dir, subfolder_name, f"{prefix}_params/", f"{model_name}.pkl"))
    
    
    def save_train_eval_results(self, model_name, results, prefix=""):
        param_dir = osp.join(self.save_dir, self.get_subfolder_name(iou_thresh=self.iou_thresh, score_thresh=self.score_thresh), f"{prefix}_train_eval_results/")
        os.makedirs(param_dir, exist_ok=True)

        #save your model or results
        joblib.dump(results, osp.join(param_dir, f"{model_name}.pkl"))
        
    def load_train_eval_results(self, model_name, prefix=""):
        subfolder_name = self.get_subfolder_name(iou_thresh=self.iou_thresh, score_thresh=self.score_thresh)
        return joblib.load(osp.join(self.save_dir, subfolder_name, f"{prefix}_train_eval_results/", f"{model_name}.pkl"))
    