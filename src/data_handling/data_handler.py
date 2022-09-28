import os
import sys
import traceback

import pandas as pd

from os import path as osp

from . import ConfigChecker

class DataHandler(ConfigChecker):
    def __init__(self, path_cfg = None, data_cfg = None, models_cfg = None):
        super(DataHandler, self).__init__(path_cfg=path_cfg, data_cfg=data_cfg, models_cfg=models_cfg)
        
        self.pred_proposal_df: pd.DataFrame = pd.DataFrame({}, columns=self.data_cfg.PRED_PROPOSAL_FIELDS)
        self.pred_output_df: pd.DataFrame = pd.DataFrame({}, columns=self.data_cfg.PRED_OUTPUT_FIELDS)
        self.gt_df: pd.DataFrame = pd.DataFrame({}, columns=self.data_cfg.GT_FIELDS)
        self.metrics_df: pd.DataFrame = pd.DataFrame({}, columns=self.data_cfg.MD3D_METRICS + self.data_cfg.LABEL_METRICS)
        
        self.X = None
        self.y_reg = None
        self.y_cls = None
        
        self.gt_path = ""
        self.pred_proposal_path = ""
        self.pred_output_path = ""
        
        self.train_img_sets = []
        self.val_img_sets = []
        self.test_img_sets = []
        self.img_sets = [] # all image sets
        self.get_train_val_test_split_filenames()
        
#         self.pred_proposal_paths = {}
#         self.pred_output_paths = {}
        self.metrics_paths = {}
                        
        self.init_paths()
#         self.load_dfs()

    @staticmethod
    def get_subfolder_name(iou_thresh, score_thresh):
        return f"iou_thresh_{iou_thresh}_score_thresh_{score_thresh}"

    def init_paths(self) -> None:
        if "kitti" == self.data_cfg.dataset_name.lower():
            velo_prefix = "reduced_" if self.data_cfg.reduced_velodyne else ""
            
            self.gt_path = osp.join(self.data_cfg.gt_dir, f"md3d_gt_input_{velo_prefix}velodyne.pkl")
            self.pred_proposal_path = osp.join(self.data_cfg.pred_dir, f"md3d_pred_proposal_input_{velo_prefix}velodyne.pkl")
            self.pred_output_path = osp.join(self.data_cfg.pred_dir, f"md3d_pred_output_input_{velo_prefix}velodyne.pkl")
        elif "nuscenes" == self.data_cfg.dataset_name.lower() or "aptiv" == self.data_cfg.dataset_name.lower():
            self.gt_path = osp.join(self.data_cfg.gt_dir, f"md3d_gt_input.pkl")  
            self.pred_proposal_path = osp.join(self.data_cfg.pred_dir, f"md3d_pred_proposal_input.pkl")
            self.pred_output_path = osp.join(self.data_cfg.pred_dir, f"md3d_pred_output_input.pkl")
        
        for score_thresh in self.data_cfg.meta_metrics_score_thresholds:
            for iou_thresh in self.data_cfg.meta_metrics_iou_thresholds:
                subfolder = self.get_subfolder_name(iou_thresh=iou_thresh, score_thresh=score_thresh)
                cur_dir = osp.join(self.data_cfg.pred_dir, f"{subfolder}/")

                if not osp.exists(cur_dir):
                    os.makedirs(cur_dir)
                
                if "kitti" == self.data_cfg.dataset_name.lower():
                    self.metrics_paths[subfolder] = osp.join(
                        self.data_cfg.pred_dir,
                        f"{subfolder}/",
                        f"md3d_metrics_{velo_prefix}velodyne.pkl"
                    )
                elif "nuscenes" == self.data_cfg.dataset_name.lower() or "aptiv" == self.data_cfg.dataset_name.lower():
                    self.metrics_paths[subfolder] = osp.join(
                        self.data_cfg.pred_dir,
                        f"{subfolder}/",
                        f"md3d_metrics.pkl"
                    ) 
                
#         self.pred_proposal_paths[subfolder] = osp.join(cur_dir, f"md3d_pred_proposal_input_{velo_prefix}velodyne.pkl")
#         self.pred_output_paths[subfolder] = osp.join(cur_dir, f"md3d_pred_output_input_{velo_prefix}velodyne.pkl")
                
    
    
    def load_dfs(self, iou_thresh, score_thresh) -> None:
        # Get crawled prediction data if exist
        self.load_pred_proposal_df(iou_thresh=iou_thresh, score_thresh=score_thresh)
        self.load_pred_output_df(iou_thresh=iou_thresh, score_thresh=score_thresh)
#         self.load_pred_proposal_df(iou_thresh=iou_thresh, score_thresh=score_thresh)
#         self.load_pred_output_df(iou_thresh=iou_thresh, score_thresh=score_thresh)
            
        # Get Ground Truth data
        if osp.exists(self.gt_path):
            self.load_gt_df()
            
        # Get MetaDetect3D Metrics if exist
        self.load_metrics_df(iou_thresh=iou_thresh, score_thresh=score_thresh)
        
        self.remove_missing_files_from_pred_output_df()
        
    def remove_missing_files_from_pred_output_df(self) -> None:
        """
        Function Checks if metrics_df and pred_output_df are well ordered and contain reasonable elements
        """
        cnt = 0
        #missing_paths = []
        ilocs = []
        for e, (x, y, z) in enumerate(zip(self.pred_output_df["x_center"], self.pred_output_df["y_center"], self.pred_output_df["z_center"])):
            if ((x < self.metrics_df["x_center_min"].iloc[e - cnt] or x > self.metrics_df["x_center_max"].iloc[e - cnt]) or 
                (y < self.metrics_df["y_center_min"].iloc[e - cnt] or y > self.metrics_df["y_center_max"].iloc[e - cnt]) or 
                (z < self.metrics_df["z_center_min"].iloc[e - cnt] or z > self.metrics_df["z_center_max"].iloc[e - cnt])):
                cnt += 1
                #missing_paths.append(self.pred_output_df["file_path"].iloc[e])
            else:
                ilocs.append(e)
        
        #missing_paths = list(set(missing_paths))
        #self.pred_output_df = self.pred_output_df[~self.pred_output_df["file_path"].isin(missing_paths)]
        self.pred_output_df = self.pred_output_df.iloc[ilocs]
        self.pred_output_df.reset_index(inplace=True, drop=True)

    def load_pred_proposal_df(self, iou_thresh=None, score_thresh=None) -> None:
        if iou_thresh is not None and score_thresh is not None:
#             path = self.pred_proposal_paths[self.get_subfolder_name(iou_thresh=iou_thresh, score_thresh=score_thresh)]
#             cur_dir = "/".join(path.split("/")[:-1])

#             if osp.exists(cur_dir):
#                 self.pred_proposal_df = pd.read_pickle(path)
#                 self.pred_proposal_df = self.pred_proposal_df[self.pred_proposal_df["score"] > score_thresh]
#             else:
#                 print(f"Directory: {cur_dir} does not exist, please check the provided thresholds.")
            self.pred_proposal_df = pd.read_pickle(self.pred_proposal_path)
            self.pred_proposal_df = self.pred_proposal_df[self.pred_proposal_df["max_score"] > score_thresh]
        else:
            self.pred_proposal_df = pd.read_pickle(self.pred_proposal_path)
    
    
    def load_pred_output_df(self, iou_thresh=None, score_thresh=None) -> None:
        if iou_thresh is not None and score_thresh is not None:
#             path = self.pred_output_paths[self.get_subfolder_name(iou_thresh=iou_thresh, score_thresh=score_thresh)]
#             cur_dir = "/".join(path.split("/")[:-1])

#             if osp.exists(cur_dir):
#                 self.pred_output_df = pd.read_pickle(path)
#                 self.pred_output_df = self.pred_output_df[self.pred_output_df["score"] > score_thresh]
#             else:
#                 print(f"Directory: {cur_dir} does not exist, please check the provided thresholds.")
            self.pred_output_df = pd.read_pickle(self.pred_output_path)
            self.pred_output_df = self.pred_output_df[self.pred_output_df["score"] > score_thresh]
        else:
            self.pred_output_df = pd.read_pickle(self.pred_output_path)
        
        
    def load_gt_df(self) -> None:
        self.gt_df = pd.read_pickle(self.gt_path)
        
        
    def load_metrics_df(self, iou_thresh=None, score_thresh=None) -> None:
        path = self.metrics_paths[self.get_subfolder_name(iou_thresh=iou_thresh, score_thresh=score_thresh)]
        cur_dir = "/".join(path.split("/")[:-1])

        if osp.exists(cur_dir):
            self.metrics_df = pd.read_pickle(path)
        else:
            print(f"Directory: {cur_dir} does not exist, please check the provided thresholds.")        

            
    def save_pred_proposal_df(self, iou_thresh=None, score_thresh=None) -> None:
        if iou_thresh is not None and score_thresh is not None:
            path = self.pred_proposal_paths[self.get_subfolder_name(iou_thresh=iou_thresh, score_thresh=score_thresh)]
            cur_dir = "/".join(path.split("/")[:-1])

            if osp.exists(cur_dir):
                self.pred_proposal_df.to_pickle(path)
            else:
                print(f"Directory: {cur_dir} does not exist, please check the provided thresholds.")
        else:
            self.pred_proposal_df.to_pickle(self.pred_proposal_path)
            
    
    def save_pred_output_df(self, iou_thresh=None, score_thresh=None) -> None:
        if iou_thresh is not None and score_thresh is not None:
            path = self.pred_output_paths[self.get_subfolder_name(iou_thresh=iou_thresh, score_thresh=score_thresh)]
            cur_dir = "/".join(path.split("/")[:-1])

            if osp.exists(cur_dir):
                self.pred_output_df.to_pickle(path)
            else:
                print(f"Directory: {cur_dir} does not exist, please check the provided thresholds.")
        else:
            self.pred_output_df.to_pickle(self.pred_output_path)
            
    
    def save_gt_df(self) -> None:
        self.gt_df.to_pickle(self.gt_path)
    
    
    def save_metrics_df(self, iou_thresh=None, score_thresh=None) -> None:
        subfolder = self.get_subfolder_name(iou_thresh=iou_thresh, score_thresh=score_thresh)
        path = self.metrics_paths[subfolder]
        cur_dir = "/".join(path.split("/")[:-1])
        
        if osp.exists(cur_dir):
            self.metrics_df.columns = self.data_cfg.MD3D_METRICS + self.data_cfg.LABEL_METRICS
            self.metrics_df.to_pickle(path)
        else:
            print(f"Directory: {cur_dir} does not exist, please check the provided thresholds.")        
    
            
    def get_train_val_test_split_filenames(self):
        if "kitti" == self.data_cfg.dataset_name.lower():
            with open(self.path_cfg.img_sets_train_path, "r") as f:
                self.train_img_sets = [line.rstrip('\n').split(" ")[0] for line in f]

            with open(self.data_cfg.img_sets_val_path, "r") as f:
                self.val_img_sets = [line.rstrip('\n').split(" ")[0] for line in f]

            if self.data_cfg.img_sets_test_path is not None:
                with open(self.data_cfg.img_sets_test_path, "r") as f:
                    self.test_img_sets = [line.rstrip('\n').split(" ")[0] for line in f]
            else:
                self.test_img_sets = []

            self.img_sets = [*self.train_img_sets, *self.val_img_sets, *self.test_img_sets]
        elif "nuscenes" == self.data_cfg.dataset_name.lower() or "aptiv" == self.data_cfg.dataset_name.lower():
            self.train_img_sets = self.data_cfg.img_sets_val_path
            self.val_img_sets = self.data_cfg.img_sets_test_path
            self.test_img_sets = []
            
            self.img_sets = [*self.train_img_sets, *self.val_img_sets, *self.test_img_sets]
