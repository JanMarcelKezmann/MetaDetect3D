import os
import shutil
import joblib
import numpy as np
import pandas as pd

from os import path as osp
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score

from . import TrainingMetrics

class Trainer(TrainingMetrics):
    def __init__(self, path_cfg = None, data_cfg = None, models_cfg = None, save_dir = None, cv: int = 5, n_jobs: int = 30, verbose: int = 5):
        super(Trainer, self).__init__(path_cfg=path_cfg, data_cfg=data_cfg, models_cfg=models_cfg)
        
        self.X_train = None
        self.X_test = None
        self.y_train_reg = None
        self.y_test_reg = None
        self.y_train_cls = None
        self.y_test_cls = None
        
        self.save_dir: str = save_dir
        self.cv: int = cv
        self.n_jobs: int = n_jobs
        self.verbose: int = verbose
            
        self.grid_results: pd.DataFrame = pd.DataFrame({})
            
        if self.save_dir is None:
            print("'save_dir' should be given, else results will not be saved.")
        
    def get_model_input(self):
        self.X_train, self.X_val, self.X_test, self.y_train_reg, self.y_val_reg, self.y_test_reg, self.y_train_cls, self.y_val_cls, self.y_test_cls = self.get_scaled_train_val_test_split()
        
    def parameter_search(self):
        for score_thresh in self.data_cfg.meta_metrics_score_thresholds:
            for iou_thresh in self.data_cfg.meta_metrics_iou_thresholds:
                self.update_thresholds(iou_thresh=iou_thresh, score_thresh=score_thresh)
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
                        self.save_grid_results(model_name, results)
#                 self.save_params()


    def train_eval(self):
        for score_thresh in self.data_cfg.meta_metrics_score_thresholds:
            for iou_thresh in self.data_cfg.meta_metrics_iou_thresholds:
                self.update_thresholds(iou_thresh=iou_thresh, score_thresh=score_thresh)
                self.get_model_input()
                
                for model_name, model_config in self.models_cfg.MODELS_CONFIG.items():
                    model = model_config["model"].__class__()
                    params = model_config["params"]
                    pipeline = model_config["pipeline"]
                    model_type = model_config["type"]
#                     scoring = model_config["scores"]
#                     refit = model_config["refit"]

                    cv_results = self.load_grid_results(model_name)["cv_results"]
                    if model_type == "regression":
                        best_params = np.asarray(cv_results["params"])[np.argsort(cv_results["rank_test_R2"])[0]]
                    else:
                        best_params = np.asarray(cv_results["params"])[np.argsort(cv_results["rank_test_Avg Precision"])[0]]

                    if pipeline is None:
#                             for param in model.get_params().keys():
#                                 if "n_jobs" in param:
#                                     model.set_params(**{param: 20})
                        model.set_params(**best_params)

                        print(model)

                        if model_type == "regression":
                            model.fit(self.X_train, self.y_train_reg["tp_score"])
                            results = {
                                "train_r2": round(model.score(self.X_train, self.y_train_reg["tp_score"]), 3),
                                "val_r2": round(model.score(self.X_val, self.y_val_reg["tp_score"]), 3),
                                "test_r2": round(model.score(self.X_test, self.y_test_reg["tp_score"]), 3) if self.X_test is not None else None,
                                "y_train_pred_reg": model.predict(self.X_train),
                                "y_val_pred_reg": model.predict(self.X_val),
                                "y_test_pred_reg": model.predict(self.X_test) if self.X_test is not None else None,
                            }
                        else:
                            model.fit(self.X_train, self.y_train_cls["tp_label"])
                            try:
                                y_train_proba = model.predict_proba(self.X_train)
                                y_val_proba = model.predict_proba(self.X_val)
                                y_test_proba = model.predict_proba(self.X_test) if self.X_test is not None else None
                            except:
                                y_train_proba, y_val_proba, y_test_proba = None, None, None
                            y_train = model.predict(self.X_train)
                            y_val = model.predict(self.X_val)
                            y_test = model.predict(self.X_test) if self.X_test is not None else None
                            results = {
                                "train_avg_precision": round(average_precision_score(y_train, self.y_train_cls["tp_label"]), 3),
                                "val_avg_precision": round(average_precision_score(y_val, self.y_val_cls["tp_label"]), 3),
                                "test_avg_precision": round(average_precision_score(y_test, self.y_test_cls["tp_label"]), 3) if self.X_test is not None else None,
                                "train_f1_macro": round(f1_score(y_train, self.y_train_cls["tp_label"], average="macro"), 3),
                                "val_f1_macro": round(f1_score(y_val, self.y_val_cls["tp_label"], average="macro"), 3),
                                "test_f1_macro": round(f1_score(y_test, self.y_test_cls["tp_label"], average="macro"), 3) if self.X_test is not None else None,
                                "train_roc_auc": round(roc_auc_score(y_train, self.y_train_cls["tp_label"]), 3),
                                "val_roc_auc": round(roc_auc_score(y_val, self.y_val_cls["tp_label"]), 3),
                                "test_roc_auc": round(roc_auc_score(y_test, self.y_test_cls["tp_label"]), 3) if self.X_test is not None else None,
                                "y_train_pred_proba_cls": y_train_proba,
                                "y_val_pred_proba_cls": y_val_proba,
                                "y_test_pred_proba_cls": y_test_proba,
                                "y_train_pred_cls": y_train,
                                "y_val_pred_cls": y_val,
                                "y_test_pred_cls": y_test,
                            
                            }

                    else:
                        poly_params = {k: v[0] for k, v in params.items() if "poly" in k}

#                             for param in pipeline.get_params().keys():
#                                 if "n_jobs" in param:
#                                     pipeline.set_params(**{param: 20})

                        pipeline.set_params(**best_params)
                        pipeline.set_params(**poly_params)

                        print(pipeline)

                        if model_type == "regression":
                            pipeline.fit(self.X_train, self.y_train_reg["tp_score"])
                            results = {
                                "train_r2": round(pipeline.score(self.X_train, self.y_train_reg["tp_score"]), 3),
                                "val_r2": round(pipeline.score(self.X_val, self.y_tval_reg["tp_score"]), 3),
                                "test_r2": round(pipeline.score(self.X_test, self.y_test_reg["tp_score"]), 3) if len(self.X_test) > 0 else None,
                                "y_train_pred_reg": model.predict(self.X_train),
                                "y_val_pred_reg": model.predict(self.X_val),
                                "y_test_pred_reg": model.predict(self.X_test) if self.X_test is not None else None,
                            }
                        else:
                            pipeline.fit(self.X_train, self.y_train_cls["tp_label"])
                            try:
                                y_train_proba = pipeline.predict_proba(self.X_train)
                                y_val_proba = pipeline.predict_proba(self.X_val)
                                y_test_proba = pipeline.predict_proba(self.X_test) if self.X_test is not None else None
                            except:
                                y_train_proba, y_val_proba, y_test_proba = None, None, None
                            y_train = pipeline.predict(self.X_train)
                            y_val = pipeline.predict(self.X_val)
                            y_test = pipeline.predict(self.X_test) if self.X_test is not None else None
                            results = {
                                "train_avg_precision": round(average_precision_score(y_train, self.y_train_cls["tp_label"]), 3),
                                "val_avg_precision": round(average_precision_score(y_val, self.y_val_cls["tp_label"]), 3),
                                "test_avg_precision": round(average_precision_score(y_test, self.y_test_cls["tp_label"]), 3) if self.X_test is not None else None,
                                "train_f1_macro": round(f1_score(y_train, self.y_train_cls["tp_label"], average="macro"), 3),
                                "val_f1_macro": round(f1_score(y_val, self.y_val_cls["tp_label"], average="macro"), 3),
                                "test_f1_macro": round(f1_score(y_test, self.y_test_cls["tp_label"], average="macro"), 3) if self.X_test is not None else None,
                                "train_roc_auc": round(roc_auc_score(y_train, self.y_train_cls["tp_label"]), 3),
                                "val_roc_auc": round(roc_auc_score(y_val, self.y_val_cls["tp_label"]), 3),
                                "test_roc_auc": round(roc_auc_score(y_test, self.y_test_cls["tp_label"]), 3) if self.X_test is not None else None,
                                "y_train_pred_proba_cls": y_train_proba,
                                "y_val_pred_proba_cls": y_val_proba,
                                "y_test_pred_proba_cls": y_test_proba,
                                "y_train_pred_cls": y_train,
                                "y_val_pred_cls": y_val,
                                "y_test_pred_cls": y_test,
                            }

                    if self.save_dir is not None:
                        self.save_train_eval_results(model_name, results)
    
    
    def save_grid_results(self, model_name, results):
#         for model_name, model_config in self.models_cfg.MODELS_CONFIG.items():
            # change this with subfolder getter
        param_dir = osp.join(self.save_dir, self.get_subfolder_name(iou_thresh=self.iou_thresh, score_thresh=self.score_thresh), "params/")
        os.makedirs(param_dir, exist_ok=True)

        #save your model or results
        joblib.dump(results, osp.join(param_dir, f"{model_name}.pkl"))
    
    
    def load_grid_results(self, model_name):
        subfolder_name = self.get_subfolder_name(iou_thresh=self.iou_thresh, score_thresh=self.score_thresh)
        return joblib.load(osp.join(self.save_dir, subfolder_name, "params/", f"{model_name}.pkl"))
    
    
    def save_train_eval_results(self, model_name, results):
        param_dir = osp.join(self.save_dir, self.get_subfolder_name(iou_thresh=self.iou_thresh, score_thresh=self.score_thresh), "train_eval_results/")
        os.makedirs(param_dir, exist_ok=True)

        #save your model or results
        joblib.dump(results, osp.join(param_dir, f"{model_name}.pkl"))
        
    def load_train_eval_results(self, model_name):
        subfolder_name = self.get_subfolder_name(iou_thresh=self.iou_thresh, score_thresh=self.score_thresh)
        return joblib.load(osp.join(self.save_dir, subfolder_name, "train_eval_results/", f"{model_name}.pkl"))
    