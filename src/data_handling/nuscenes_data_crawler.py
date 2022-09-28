import os

import numpy as np
import pandas as pd

from tqdm import tqdm
from os import path as osp
from collections import Counter

from mmdet3d.apis import init_model
from mmdet3d.datasets import build_dataset
from mmdet3d.datasets.pipelines import Compose
from mmdet3d.core.bbox import box_np_ops

from . import DataHandler
from ..od_inference_detectors import *


class NuScenesDataCrawler(DataHandler):
    def __init__(self, path_cfg = None, data_cfg = None, models_cfg = None, device = None):
        super(NuScenesDataCrawler, self).__init__(path_cfg=path_cfg, data_cfg=data_cfg, models_cfg=models_cfg)
                    
#         self.train_velo_dir = self.path_cfg.NUSCENES_TRAIN_VELO_DIR
#         self.train_velo_dir = osp.join(self.path_cfg.NUSCENES_SAMPLES_DIR, "LIDAR_TOP/")
    
        self.nuscenes_infos_val = pd.read_pickle(self.data_cfg.infos_val_path)["infos"]
        self.val_lidar_files = [info["lidar_path"] for info in self.nuscenes_infos_val]

        self.model = init_model(self.data_cfg.model_cfg_path, self.data_cfg.model_ckpt_path, device)
        

    def crawl_pred(self) -> None:        
#         for score_thresh in self.data_cfg.od_score_thresholds:
#             for iou_thresh in self.data_cfg.od_iou_thresholds:
#                 print(self.get_subfolder_name(iou_thresh=iou_thresh, score_thresh=score_thresh))
        temp_pred_output_df = pd.DataFrame({}, columns=self.data_cfg.PRED_OUTPUT_FIELDS)
        temp_pred_proposal_df = pd.DataFrame({}, columns=self.data_cfg.PRED_PROPOSAL_FIELDS)
        for filename in tqdm(sorted(self.val_lidar_files), total=len(self.val_lidar_files)):
            if filename.split("/")[-1].split("__")[0] in self.img_sets:
                #if filename.split(".")[0] in self.val_img_sets or filename.split(".")[0] in self.test_img_sets:
                filepath = osp.join(self.path_cfg.MMDET3D_DIR, "/".join(filename.split("/")[1:]))
                pred_output_df, pred_proposal_df = self.get_pred_labels(filepath)

                temp_pred_output_df = pd.concat([temp_pred_output_df, pred_output_df], ignore_index=True)
                temp_pred_proposal_df = pd.concat([temp_pred_proposal_df, pred_proposal_df], ignore_index=True)

                if len(temp_pred_proposal_df) % 10000 == 0:
                    self.pred_output_df = pd.concat([self.pred_output_df, temp_pred_output_df], ignore_index=True)
                    temp_pred_output_df = pd.DataFrame({}, columns=self.data_cfg.PRED_OUTPUT_FIELDS)

                    self.pred_proposal_df = pd.concat([self.pred_proposal_df, temp_pred_proposal_df], ignore_index=True)
                    temp_pred_proposal_df = pd.DataFrame({}, columns=self.data_cfg.PRED_PROPOSAL_FIELDS)


        self.pred_output_df = pd.concat([self.pred_output_df, temp_pred_output_df], ignore_index=True)
        self.pred_proposal_df = pd.concat([self.pred_proposal_df, temp_pred_proposal_df], ignore_index=True)

#                 self.save_pred_proposal_df(iou_thresh=iou_thresh, score_thresh=score_thresh)
#                 self.save_pred_output_df(iou_thresh=iou_thresh, score_thresh=score_thresh)

    def get_pred_labels(self, filepath):
        # make the inference detector general for all models
        output, proposals, data, intensities = nuscenes_inference_detector(self.model, filepath)
        
        output_rows, proposal_rows = [], []
        for num, (box_3d, scores, dir_score) in enumerate(zip(proposals[0]["boxes_3d"], proposals[0]["scores_3d"], proposals[0]["dir_scores_3d"])):
            pib = proposals[0]["boxes_3d"][num].points_in_boxes(data["points"][0][0][:, :3]).cpu().numpy() + 1
            pib_num = Counter(pib)[1]
            pib_reflectance = intensities[pib.astype(bool)]
            
            assert pib_num == len(pib_reflectance), print(f"{filepath}: Number of points in box does not match pib reflectance array length.")
            
            if ("centerpoint" in str(self.model.__class__).lower() and pib_num < 5):
                continue

            if len(pib_reflectance) == 0:
                pib_reflectance = [0]
                
            if "centerpoint" in str(self.model.__class__).lower():
                row = [
                    osp.join(filepath),                             # file_path
                    *box_3d.cpu().numpy(),                          # x_center, y_center, z_center, length, width, height, theta
                    np.amax(scores.cpu().numpy()),                  # max_score
                    np.argmax(scores.cpu().numpy()),                # category_idx
                    pib_num,                                        # num_pib
                    pib_num / float(data["points"][0][0].shape[0]), # num_pib divided by total number of points in point cloud
                    np.max(pib_reflectance),                        # max reflectance in pib
                    np.mean(pib_reflectance),                       # mean reflectance in pib
                    np.std(pib_reflectance),                        # std reflectance in pib
                    num                                             # dataset_box_id
                ]
            else:
                row = [
                    osp.join(filepath),                             # file_path
                    *box_3d.cpu().numpy(),                          # x_center, y_center, z_center, length, width, height, theta
                    dir_score.cpu().item(),                         # dir_score
                    np.amax(scores.cpu().numpy()),                  # max_score
                    np.sum(scores.cpu().numpy()),                   # score_sum
                    *scores.cpu().numpy(),                          # score_0, score_1, ..., score_9
                    np.argmax(scores.cpu().numpy()),                # category_idx
                    pib_num,                                        # num_pib
                    pib_num / float(data["points"][0][0].shape[0]), # num_pib divided by total number of points in point cloud
                    np.max(pib_reflectance),                        # max reflectance in pib
                    np.mean(pib_reflectance),                       # mean reflectance in pib
                    np.std(pib_reflectance),                        # std reflectance in pib
                    num                                             # dataset_box_id
                ]

            assert len(self.data_cfg.PRED_PROPOSAL_FIELDS) == len(row), print(f"{filepath}: Number of prediction df columns does not match length of row.")
            
            proposal_rows.append(row)

        for num, (box_3d, score, label, dir_score) in enumerate(zip(output[0]["boxes_3d"], output[0]["scores_3d"], output[0]["labels_3d"], output[0]["dir_scores_3d"])):
            pib = output[0]["boxes_3d"][num].points_in_boxes(data["points"][0][0][:, :3]).cpu().numpy() + 1
#             pib_reflectance = data["points"][0][0][pib.astype(bool)][:, 3].cpu().numpy()
            pib_reflectance = intensities[pib.astype(bool)]
            if len(pib_reflectance) == 0:
                pib_reflectance = [0]
            pib_num = Counter(pib)[1]
            
            if ("centerpoint" in str(self.model.__class__).lower() and pib_num < 5):
                continue

            if "centerpoint" in str(self.model.__class__).lower():
                row = [
                    osp.join(filepath),                             # file_path
                    *box_3d.cpu().numpy(),                          # x_center, y_center, z_center, length, width, height, theta
                    score.cpu().item(),                             # score
                    label.cpu().item(),                             # category_idx
                    pib_num,                                        # num_pib
                    pib_num / float(data["points"][0][0].shape[0]), # num_pib divided by total number of points in point cloud
                    np.max(pib_reflectance),                        # max reflectance in pib
                    np.mean(pib_reflectance),                       # mean reflectance in pib
                    np.std(pib_reflectance),                        # std reflectance in pib
                    num                                             # dataset_box_id
                ]
            else:
                row = [
                    osp.join(filepath),                             # file_path
                    *box_3d.cpu().numpy(),                          # x_center, y_center, z_center, length, width, height, theta
                    dir_score.cpu().item(),                         # dir_score
                    score.cpu().item(),                             # score
                    label.cpu().item(),                             # category_idx
                    pib_num,                                        # num_pib
                    pib_num / float(data["points"][0][0].shape[0]), # num_pib divided by total number of points in point cloud
                    np.max(pib_reflectance),                        # max reflectance in pib
                    np.mean(pib_reflectance),                       # mean reflectance in pib
                    np.std(pib_reflectance),                        # std reflectance in pib
                    num                                             # dataset_box_id
                ]
            
            assert len(self.data_cfg.PRED_OUTPUT_FIELDS) == len(row), print(f"{filename}: Number of prediction df columns does not match length of row.")

            output_rows.append(row)

        return pd.DataFrame(output_rows, columns=self.data_cfg.PRED_OUTPUT_FIELDS), pd.DataFrame(proposal_rows, columns=self.data_cfg.PRED_PROPOSAL_FIELDS)
    
    
    def crawl_gt(self) -> None:
#         dataset = self.build_gt_dataset()
        labels = []
        
        known_labels = []
        unknown_labels = []
    
        for filenum in tqdm(range(len(self.val_lidar_files)), total=len(self.val_lidar_files)):
            
            filename = self.val_lidar_files[filenum]
            gt_boxes = self.nuscenes_infos_val[filenum]["gt_boxes"]
            gt_names = self.nuscenes_infos_val[filenum]["gt_names"]
            num_lidar_pts = self.nuscenes_infos_val[filenum]["num_lidar_pts"]
            
            for enum, (box, label, num_pts) in enumerate(zip(gt_boxes, gt_names, num_lidar_pts)):
                if ("centerpoint" in str(self.model.__class__).lower() and num_pts >= 5) or ("mvxfasterrcnn" in str(self.model.__class__).lower()):
                    if label in self.data_cfg.class_to_num.keys():
                        box3d = [osp.join(self.path_cfg.MMDET3D_DIR, "/".join(filename.split("/")[1:]))] + list(box)
                        box3d.append(self.data_cfg.class_to_num[label])
                        box3d.append(num_pts)
                        box3d.append(enum)

                        assert len(self.data_cfg.GT_FIELDS) == len(box3d), print(f"{filename}: Number of prediction df columns does not match length of box3d.")

                        labels.append(box3d)
                        known_labels.append(label)
                    else:
                        unknown_labels.append(label)
        
        self.gt_df = pd.DataFrame(labels, columns=self.data_cfg.GT_FIELDS)
#         return known_labels, unknown_labels
        