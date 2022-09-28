import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

from .. import DataHandler

class TrainingMetrics(DataHandler):
    def __init__(self, path_cfg = None, data_cfg = None, models_cfg = None, iou_thresh = None, score_thresh = None, dtype = np.float64):
        super(TrainingMetrics, self).__init__(path_cfg=path_cfg, data_cfg=data_cfg, models_cfg=models_cfg)
        
        self.dtype = dtype
                
        self.model = None
        
        self.iou_thresh = iou_thresh
        self.score_thresh = score_thresh
        
#         if self.iou_thresh is not None and self.score_thresh is not None:
#             self.load_dfs(iou_thresh=iou_thresh, score_thresh=score_thresh)
        
#         self.get_labels()
#         self.get_input()        
    
    def load_input_and_labels(self, iou_thresh=None, score_thresh=None):
        self.update_thresholds(iou_thresh=iou_thresh, score_thresh=score_thresh)
        
        self.load_dfs(iou_thresh=self.iou_thresh, score_thresh=self.score_thresh)
        
        self.get_labels()
        self.get_input()

        
    def update_thresholds(self, iou_thresh=None, score_thresh=None):
        if iou_thresh is not None:
            self.iou_thresh = iou_thresh
        if score_thresh is not None:
            self.score_thresh = score_thresh
  

    def get_labels(self):
        self.y_reg = self.metrics_df[self.data_cfg.LABEL_METRICS]
        self.y_reg.columns = ["tp_score", "dataset_box_id"]
        
#         if self.data_cfg.meta_model in self.models_cfg.CLASSIFICATION_MODELS:
#         labels = np.array(self.y_reg["tp_score"]) > self.data_cfg.md_iou_threshold
        labels = np.array(self.y_reg["tp_score"]) > self.iou_thresh
        self.y_cls = self.y_reg.copy()
        self.y_cls.columns = ["tp_label", "dataset_box_id"]
        self.y_cls["tp_label"] = labels
    
    
    def get_input(self):
#         self.X = pd.concat([self.pred_output_df[self.data_cfg.OUTPUT_METRICS], self.metrics_df], axis=1)
        self.X = pd.concat([self.pred_output_df[self.data_cfg.PRED_OUTPUT_FIELDS].reset_index(), self.metrics_df[self.data_cfg.MD3D_METRICS]], axis=1)
        
        self.X.drop(columns=["file_path", "dataset_box_id"], inplace=True)
    
    
    def get_train_val_test_split(self, iou_thresh=None, score_thresh=None):
        self.update_thresholds(iou_thresh=iou_thresh, score_thresh=score_thresh)

        self.load_input_and_labels()
        
        train_mask, val_mask, test_mask = [], [], []
        for file_path in tqdm(self.pred_output_df["file_path"], total=len(self.pred_output_df["file_path"])):
            if file_path.split("/")[-1].split(".")[0] in self.train_img_sets:
                train_mask.append(True)
                val_mask.append(False)
                test_mask.append(False)
            elif file_path.split("/")[-1].split(".")[0] in self.val_img_sets:
                train_mask.append(False)
                val_mask.append(True)
                test_mask.append(False)
            else:
                train_mask.append(False)
                val_mask.append(False)
                test_mask.append(True)
                
        return self.X[train_mask], self.X[val_mask], self.X[test_mask], self.y_reg[train_mask], self.y_reg[val_mask], self.y_reg[test_mask], self.y_cls[train_mask], self.y_cls[val_mask], self.y_cls[test_mask]
    
    
    def get_scaled_train_val_test_split(self, iou_thresh=None, score_thresh=None):
        self.update_thresholds(iou_thresh=iou_thresh, score_thresh=score_thresh)
                                    
        X_train, X_val, X_test, y_train_reg, y_val_reg, y_test_reg, y_train_cls, y_val_cls, y_test_cls = self.get_train_val_test_split()
        
        scaler = StandardScaler().fit(X_train)
        
        if len(X_test) == 0:
            return scaler.transform(X_train), scaler.transform(X_val), None, y_train_reg, y_val_reg, None, y_train_cls, y_val_cls, None
        else:
            return scaler.transform(X_train), scaler.transform(X_val), scaler.transform(X_test), y_train_reg, y_val_reg, y_test_reg, y_train_cls, y_val_cls, y_test_cls