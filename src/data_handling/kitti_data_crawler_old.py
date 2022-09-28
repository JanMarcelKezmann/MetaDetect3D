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


class KITTIDataCrawlerOld(DataHandler):
    def __init__(self, path_cfg = None, data_cfg = None, models_cfg = None):
        super(KITTIDataCrawler, self).__init__(path_cfg=path_cfg, data_cfg=data_cfg, models_cfg=models_cfg)
                    
        if self.data_cfg.reduced_velodyne:
            self.train_velo_dir = self.path_cfg.KITTI_TRAIN_VELO_RED_DIR
        else:
            self.train_velo_dir = self.path_cfg.KITTI_TRAIN_VELO_DIR
        
        self.model = None
                

    def crawl_pred(self, device) -> None:
        # TODO make this general for all models
        self.model = init_model(self.data_cfg.model_cfg_path, self.data_cfg.model_ckpt_path, device)
        
        temp_pred_output_df = pd.DataFrame({}, columns=self.data_cfg.PRED_OUTPUT_FIELDS)
        temp_pred_proposal_df = pd.DataFrame({}, columns=self.data_cfg.PRED_PROPOSAL_FIELDS)
        for filename in tqdm(sorted(os.listdir(self.train_velo_dir)), total=len(os.listdir(self.train_velo_dir))):
            if filename.split(".")[0] in self.img_sets:
                pred_output_df, pred_proposal_df = self.get_pred_labels(filename)

                temp_pred_output_df = pd.concat([temp_pred_output_df, pred_output_df], ignore_index=True)
                temp_pred_proposal_df = pd.concat([temp_pred_proposal_df, pred_proposal_df], ignore_index=True)

                if len(temp_pred_proposal_df) % 10000 == 0:
                    self.pred_output_df = pd.concat([self.pred_output_df, temp_pred_output_df], ignore_index=True)
                    temp_pred_output_df = pd.DataFrame({}, columns=self.data_cfg.PRED_OUTPUT_FIELDS)

                    self.pred_proposal_df = pd.concat([self.pred_proposal_df, temp_pred_proposal_df], ignore_index=True)
                    temp_pred_proposal_df = pd.DataFrame({}, columns=self.data_cfg.PRED_PROPOSAL_FIELDS)


        self.pred_output_df = pd.concat([self.pred_output_df, temp_pred_output_df], ignore_index=True)
        self.pred_proposal_df = pd.concat([self.pred_proposal_df, temp_pred_proposal_df], ignore_index=True)

    
    def get_pred_labels(self, filename):
        # make the inference detector general for all models
        output, proposals, data = second_inference_detector(self.model, osp.join(self.train_velo_dir, filename))
        
        output_rows, proposal_rows = [], []
        for num, (box_3d, scores, dir_score) in enumerate(zip(proposals[0]["boxes_3d"], proposals[0]["scores_3d"], proposals[0]["dir_scores_3d"])):
            row = [
                osp.join(self.train_velo_dir, filename), # file_path
                *box_3d.cpu().numpy(),                   # x_center, y_center, z_center, length, width, height, theta
                dir_score.cpu().item(),                  # dir_score
                np.amax(scores.cpu().numpy()),           # max_score
                np.sum(scores.cpu().numpy()),            # score_sum
                *scores.cpu().numpy(),                   # score_0, score_1, score_2
                np.argmax(scores.cpu().numpy()),         # category_idx
                Counter(proposals[0]["boxes_3d"][num].points_in_boxes(data["points"][0][0][:, :3]).cpu().numpy())[0], # pib
                num                                      # dataset_box_id
            ]
            
            assert len(self.data_cfg.PRED_PROPOSAL_FIELDS) == len(row), print(f"{filename}: Number of prediction df columns does not match length of row.")

            proposal_rows.append(row)
        
        for num, (box_3d, score, label, dir_score) in enumerate(zip(output[0]["boxes_3d"], output[0]["scores_3d"], output[0]["labels_3d"], output[0]["dir_scores_3d"])):
            row = [
                osp.join(self.train_velo_dir, filename), # file_path
                *box_3d.cpu().numpy(),                   # x_center, y_center, z_center, length, width, height, theta
                dir_score.cpu().item(),                  # dir_score
                score.cpu().item(),                      # score
                label.cpu().item(),                      # category_idx
                Counter(output[0]["boxes_3d"][num].points_in_boxes(data["points"][0][0][:, :3]).cpu().numpy())[0], # pib
                num                                      # dataset_box_id
            ]
            
            assert len(self.data_cfg.PRED_OUTPUT_FIELDS) == len(row), print(f"{filename}: Number of prediction df columns does not match length of row.")

            output_rows.append(row)
        
            
        return pd.DataFrame(output_rows, columns=self.data_cfg.PRED_OUTPUT_FIELDS), pd.DataFrame(proposal_rows, columns=self.data_cfg.PRED_PROPOSAL_FIELDS)

    
    def get_gt_dataset_cfg(self) -> dict:
        gt_dataset_cfg = dict(type="KittiDataset", data_root=self.path_cfg.KITTI_BASE_DIR, ann_file=self.path_cfg.infos_trainval_path)

        gt_dataset_cfg.update(
            test_mode=False,
            split='training',
            modality=dict(use_lidar=True, use_depth=False, use_lidar_intensity=True, use_camera=False),
            pipeline=[
                dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4, file_client_args={'backend': 'disk'}),
                dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, file_client_args={'backend': 'disk'})
            ])
        
        return gt_dataset_cfg
    
    
    def build_gt_dataset(self):
        gt_dataset_cfg = self.get_gt_dataset_cfg()
        return build_dataset(gt_dataset_cfg)
        
        
    def crawl_gt(self) -> None:
        dataset = self.build_gt_dataset()
        
        for filenum in tqdm(range(len(dataset)), total=len(dataset)):
            out = self.get_gt_train_labels(dataset, filenum)
            self.gt_df = pd.concat([self.gt_df, out], ignore_index=True)
            
        self.gt_df.columns = self.data_cfg.GT_FIELDS
        
        
    def get_gt_train_labels(self, dataset, j) -> pd.DataFrame:
        labels = []
        
        input_dict = dataset.get_data_info(j)
        
        if self.data_cfg.reduced_velodyne:
            input_dict["pts_filename"] = input_dict["pts_filename"].replace("velodyne", "velodyne_reduced")

        dataset.pre_pipeline(input_dict)
        example = dataset.pipeline(input_dict)
        
        filenum = example["pts_filename"].split("/")[-1].split(".")[0]
        
        # if is only a double check since ann_file is configured properly
        #if filenum in self.val_img_sets or filenum in self.test_img_sets:
        if filenum in self.img_sets:
            annos = example['ann_info']
            points = example['points'].tensor.numpy()
            gt_boxes_3d = annos['gt_bboxes_3d'].tensor.numpy()
            point_indices = box_np_ops.points_in_rbbox(points, gt_boxes_3d)

            for enum, label_name in enumerate(annos["gt_names"]):
                if label_name in ["Pedestrian", "Cyclist", "Car"]:                
                    box3d = [example["pts_filename"]] + list(gt_boxes_3d[enum])
                    box3d.append(self.data_cfg.class_to_num[label_name])
                    box3d.append(points[point_indices[:, enum]].shape[0])
                    box3d.append(enum)

                    # maybe save bboxes only if more than n points are in box e.g. 5

                    labels.append(box3d)

                    assert len(self.data_cfg.GT_FIELDS) == len(box3d), print(f"{filename}: Number of prediction df columns does not match length of box3d.")
                
            return pd.DataFrame(labels, columns=self.data_cfg.GT_FIELDS)
        else:
            return pd.DataFrame({}, columns=self.data_cfg.GT_FIELDS)
        