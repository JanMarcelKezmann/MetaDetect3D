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


class AptivDataCrawler(DataHandler):
    def __init__(self, path_cfg = None, data_cfg = None, models_cfg = None, device = None):
        super(AptivDataCrawler, self).__init__(path_cfg=path_cfg, data_cfg=data_cfg, models_cfg=models_cfg)
        
        self.aptiv_infos_valtest = pd.read_pickle(self.path_cfg.sdb_infos_valtest_path)
        self.valtest_lidar_files = [info["lidar_points"]["lidar_path"] for info in self.aptiv_infos_valtest]

        self.model = init_model(self.data_cfg.model_cfg_path, self.data_cfg.model_ckpt_path, device)

    def crawl_pred(self) -> None:
        temp_pred_output_df = pd.DataFrame({}, columns=self.data_cfg.PRED_OUTPUT_FIELDS)
        temp_pred_proposal_df = pd.DataFrame({}, columns=self.data_cfg.PRED_PROPOSAL_FIELDS)
        for filepath in tqdm(sorted(self.valtest_lidar_files), total=len(self.valtest_lidar_files)):
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

    
    def get_pred_labels(self, filepath):
        # make the inference detector general for all models
        output, proposals, data = aptiv_inference_detector(self.model, filepath)
                
        intensities = data["points"][0][0].cpu().numpy()[:, 3]
        
        output_rows, proposal_rows = [], []
        
        for num, (box_3d, scores, dir_score) in enumerate(zip(proposals[0]["boxes_3d"], proposals[0]["scores_3d"], proposals[0]["dir_scores_3d"])):
            pib = proposals[0]["boxes_3d"][num].points_in_boxes(data["points"][0][0][:, :3]).cpu().numpy() + 1
            pib_num = Counter(pib)[1]
            pib_reflectance = intensities[pib.astype(bool)]
            
            assert pib_num == len(pib_reflectance), print(f"{filepath}: Number of points in box does not match pib reflectance array length.")
            
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
                    *scores.cpu().numpy(),                          # score_0, score_1, ..., score_4
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
        min_points = {
            'car_or_van_or_suv_or_pickup': 5,
            'truck_or_bus': 5,
            'pedestrian': 3,
            'motorcycle_or_bicycle': 3
        }
        
        # add thresholding regarding pib for both datasets according to configs
        labels = []
        
        known_labels = []
        unknown_labels = []
        
        for filenum in tqdm(range(len(self.valtest_lidar_files)), total=len(self.valtest_lidar_files)):
            
            filename = self.valtest_lidar_files[filenum]

            gt_boxes = self.aptiv_infos_valtest[filenum]["annos"]["gt_boxes_3d"]
            gt_names = self.aptiv_infos_valtest[filenum]["annos"]["gt_names"]
            num_lidar_pts = self.aptiv_infos_valtest[filenum]["annos"]["num_points_in_gt"]
            
            for enum, (box, label, num_pts) in enumerate(zip(gt_boxes, gt_names, num_lidar_pts)):

                if min_points[label] <= num_pts:
                    box[2] = box[2] - box[5] / 2
                    box3d = [filename] + list(box)
                    box3d.append(self.data_cfg.class_to_num[label])
                    box3d.append(num_pts)
                    box3d.append(enum)

                    assert len(self.data_cfg.GT_FIELDS) == len(box3d), print(f"{filename}: Number of prediction df columns does not match length of box3d.")

                    labels.append(box3d)
                    
        self.gt_df = pd.DataFrame(labels, columns=self.data_cfg.GT_FIELDS)
#         return known_labels, unknown_labels
        