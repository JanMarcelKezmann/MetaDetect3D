import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

from .. import DataHandler

class TrainingMetrics(DataHandler):
    def __init__(self, path_cfg = None, data_cfg = None, models_cfg = None, iou_thresh = None, score_thresh = None, baseline = None, dtype = np.float64):
        super(TrainingMetrics, self).__init__(path_cfg=path_cfg, data_cfg=data_cfg, models_cfg=models_cfg)
        
        self.dtype = dtype
        
        self.model = None
        
        self.iou_thresh = iou_thresh
        self.score_thresh = score_thresh
        self.baseline = baseline
        
#         if self.iou_thresh is not None and self.score_thresh is not None:
#             self.load_dfs(iou_thresh=iou_thresh, score_thresh=score_thresh)
        
#         self.get_labels()
#         self.get_input()        
    
    def load_input_and_labels(self, iou_thresh=None, score_thresh=None, drop_path_box_id=True, baseline=None):
        self.update_thresholds(iou_thresh=iou_thresh, score_thresh=score_thresh, baseline=baseline)
        
        self.load_dfs(iou_thresh=self.iou_thresh, score_thresh=self.score_thresh)
        
        self.get_labels(drop_path_box_id=drop_path_box_id)
        self.get_input(drop_path_box_id=drop_path_box_id)

        
    def update_thresholds(self, iou_thresh=None, score_thresh=None, baseline=None):
        if iou_thresh is not None:
            self.iou_thresh = iou_thresh
        if score_thresh is not None:
            self.score_thresh = score_thresh
        if baseline is not None:
            self.baseline = baseline
  

    def get_labels(self, drop_path_box_id=True):
        self.y_reg = self.metrics_df[self.data_cfg.LABEL_METRICS]
        self.y_reg.columns = ["tp_score", "dataset_box_id"]
        #self.y_reg["score"] = self.pred_output_df["score"]
        if not drop_path_box_id:
            self.y_reg["file_path"] = self.pred_output_df["file_path"]
        
        #self.y_reg = self.y_reg[self.y_reg["score"] > self.score_thresh]
        #self.y_reg.drop(columns="score", inplace=True)
        
        
#         if self.data_cfg.meta_model in self.models_cfg.CLASSIFICATION_MODELS:
#         labels = np.array(self.y_reg["tp_score"]) > self.data_cfg.md_iou_threshold
        labels = np.array(self.y_reg["tp_score"]) > self.iou_thresh
        self.y_cls = self.y_reg.copy()
        if drop_path_box_id:
            self.y_cls.columns = ["tp_label", "dataset_box_id"]
        else:
            self.y_cls.columns = ["tp_label", "dataset_box_id", "file_path"]
        self.y_cls["tp_label"] = labels
    
    
    def get_input(self, drop_path_box_id=True):
#         self.X = pd.concat([self.pred_output_df[self.data_cfg.OUTPUT_METRICS], self.metrics_df], axis=1)
        self.X = pd.concat([self.pred_output_df[self.data_cfg.PRED_OUTPUT_FIELDS].reset_index(), self.metrics_df[self.data_cfg.MD3D_METRICS]], axis=1)
        self.X = self.X[self.X["score"] > self.score_thresh]

        if drop_path_box_id:
            self.X.drop(columns=["file_path", "dataset_box_id"], inplace=True)
            
        if "index" in self.X.columns:
            self.X.drop(columns=["index"], inplace=True)
        
        if self.baseline:
            self.X = self.X["score"]
    
    
    def get_train_val_split(self, iou_thresh=None, score_thresh=None, drop_path_box_id=True, baseline=None):
        self.update_thresholds(iou_thresh=iou_thresh, score_thresh=score_thresh, baseline=baseline)

        self.load_input_and_labels(drop_path_box_id=drop_path_box_id)
                
        train_mask, val_mask = [], []
        #for file_path in tqdm(self.pred_output_df["file_path"], total=len(self.pred_output_df["file_path"])):
        for file_path in tqdm(self.pred_output_df[self.pred_output_df["score"] > self.score_thresh]["file_path"], total=len(self.pred_output_df[self.pred_output_df["score"] > self.score_thresh]["file_path"])):
            if self.data_cfg.img_sets_test_path is None or len(self.test_img_sets) == 0:
                if self.data_cfg.dataset_name.lower() == "kitti":
                    if file_path.split("/")[-1].split(".")[0] in self.train_img_sets:
                        train_mask.append(True)
                        val_mask.append(False)
                    elif file_path.split("/")[-1].split(".")[0] in self.val_img_sets:
                        train_mask.append(False)
                        val_mask.append(True)
                    else:
                        train_mask.append(False)
                        val_mask.append(False)
                elif self.data_cfg.dataset_name.lower() == "nuscenes":
                    if file_path.split("/")[-1].split("__")[0] in self.train_img_sets:
                        train_mask.append(True)
                        val_mask.append(False)
                    elif file_path.split("/")[-1].split("__")[0] in self.val_img_sets:
                        train_mask.append(False)
                        val_mask.append(True)
                    else:
                        train_mask.append(False)
                        val_mask.append(False)
                elif self.data_cfg.dataset_name.lower() == "aptiv":
                    if file_path in self.train_img_sets:
                        train_mask.append(True)
                        val_mask.append(False)
                    elif file_path in self.val_img_sets:
                        train_mask.append(False)
                        val_mask.append(True)
                    else:
                        train_mask.append(False)
                        val_mask.append(False)
            else:
                if self.data_cfg.dataset_name.lower() == "kitti":
                    if file_path.split("/")[-1].split(".")[0] in self.val_img_sets:
                        train_mask.append(True)
                        val_mask.append(False)
                    elif file_path.split("/")[-1].split(".")[0] in self.test_img_sets:
                        train_mask.append(False)
                        val_mask.append(True)
                    else:
                        train_mask.append(False)
                        val_mask.append(False)
              
        if self.baseline:
            return np.expand_dims(self.X[train_mask], axis=1), np.expand_dims(self.X[val_mask], axis=1), self.y_reg[train_mask], self.y_reg[val_mask], self.y_cls[train_mask], self.y_cls[val_mask]
        else:
            return self.X[train_mask], self.X[val_mask], self.y_reg[train_mask], self.y_reg[val_mask], self.y_cls[train_mask], self.y_cls[val_mask]
    
    
    def get_scaled_train_val_split(self, iou_thresh=None, score_thresh=None, drop_path_box_id=True, baseline=None):
        self.update_thresholds(iou_thresh=iou_thresh, score_thresh=score_thresh, baseline=baseline)
                                    
        X_train, X_val, y_train_reg, y_val_reg, y_train_cls, y_val_cls = self.get_train_val_split(drop_path_box_id=drop_path_box_id)
                
        if drop_path_box_id:
            scaler = StandardScaler().fit(X_train)
            return scaler.transform(X_train), scaler.transform(X_val), y_train_reg, y_val_reg, y_train_cls, y_val_cls
            
        else:
            scaler = StandardScaler().fit(X_train.to_numpy()[:, 1:])
            
            X_train_scaled = scaler.transform(X_train.to_numpy()[:, 1:])
            X_val_scaled = scaler.transform(X_val.to_numpy()[:, 1:])
            
            X_train.to_numpy()[:, 1:] = X_train_scaled
            X_val.to_numpy()[:, 1:] = X_val_scaled
            
            
            return X_train.to_numpy(), X_val.to_numpy(), y_train_reg, y_val_reg, y_train_cls, y_val_cls
        