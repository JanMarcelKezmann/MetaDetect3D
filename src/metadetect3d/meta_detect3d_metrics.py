import os
import numba
import torch

import numpy as np
import pandas as pd

from tqdm import tqdm

from mmdet3d.ops.iou3d import boxes_iou_bev
from mmdet3d.core import bbox_overlaps_3d, xywhr2xyxyr
from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes

from .. import DataHandler


class MetaDetect3DMetrics(DataHandler):
    def __init__(self, path_cfg, data_cfg, models_cfg, dtype = np.float64):
        super(MetaDetect3DMetrics, self).__init__(path_cfg=path_cfg, data_cfg=data_cfg, models_cfg=models_cfg)
        
        self.dtype = dtype
        
        self.test_cfg = {
            'use_rotate_nms': True,
            'nms_across_levels': False,
            'nms_thr': 0.01,
            'score_thr': 0.1,
            'min_bbox_size': 0,
            'nms_pre': 100,
            'max_num': 50
        }
        
        self.load_gt_df()
        self.load_pred_output_df()
        self.load_pred_proposal_df()
        
        if "centerpoint" in self.data_cfg.model_cfg_path:
            self.centerpoint_bool = True
        else:
            self.centerpoint_bool = False

    def get_and_save_metrics(self) -> None:
        for score_thresh in self.data_cfg.meta_metrics_score_thresholds:
            for iou_thresh in self.data_cfg.meta_metrics_iou_thresholds:
                print(self.get_subfolder_name(iou_thresh=iou_thresh, score_thresh=score_thresh))
#                 self.proposal_bboxes = self.pred_proposal_df.loc[self.pred_proposal_df['max_score'] > self.data_cfg.od_score_threshold]
#                 self.output_bboxes = self.pred_output_df.loc[self.pred_output_df['score'] > self.data_cfg.od_score_threshold]
                self.proposal_bboxes = self.pred_proposal_df.loc[self.pred_proposal_df['max_score'] > score_thresh]
                self.output_bboxes = self.pred_output_df.loc[self.pred_output_df['score'] > score_thresh]

                all_pc_paths = sorted(list(set(self.pred_output_df["file_path"])))

                temp_metrics_df = pd.DataFrame({})
                for i, file_path in tqdm(enumerate(all_pc_paths), total=len(all_pc_paths)):
                    # Get BBoxes belonging to current file_path without file_path and dataset_box_id infos
                    self.proposal_bboxes_single_pc = self.proposal_bboxes[self.proposal_bboxes["file_path"] == file_path].to_numpy()[:, 1:-1].astype(self.dtype)
                    self.output_bboxes_single_pc = self.output_bboxes[self.output_bboxes["file_path"] == file_path].to_numpy()[:, 1:-1].astype(self.dtype)
                    self.labels_single_pc = self.gt_df[self.gt_df["file_path"] == file_path].to_numpy()[:, 1:8].astype(self.dtype)
                    
                    if len(self.output_bboxes_single_pc) == 0:
                        continue
                    
                    # Get Metrics of a single point cloud for all proposal 3d bboxes
                    try:
                        metrics_single_pc = self.get_metrics_of_single_pc(iou_thresh)
                    except:
                        print(i, file_path)
                        continue

                    # Update metrics and IoU matrix
                    temp_metrics_df = temp_metrics_df.append(pd.DataFrame(metrics_single_pc))
                    if i % 500 == 0 and i != 0:
                        temp_metrics_df.columns = self.metrics_df.columns
                        self.metrics_df = pd.concat([self.metrics_df, temp_metrics_df], ignore_index=True)
                        temp_metrics_df = pd.DataFrame({})

                if len(temp_metrics_df) > 0:
                    temp_metrics_df.columns = self.metrics_df.columns
                    self.metrics_df = pd.concat([self.metrics_df, temp_metrics_df], ignore_index=True)
                
                self.save_metrics_df(iou_thresh=iou_thresh, score_thresh=score_thresh)
                #self.save_missing_paths()
                
                self.metrics_df = pd.DataFrame({}, columns=self.data_cfg.MD3D_METRICS + self.data_cfg.LABEL_METRICS)

   
    def get_metrics_of_single_pc(self, iou_thresh) -> np.ndarray:
        output_bboxes_arr = np.array(self.output_bboxes_single_pc)
        # Metrics single pc have dim (num_output_bboxes x len(self.data_cfg.MD3D_METRICS))
        metrics_single_pc = np.zeros((output_bboxes_arr.shape[0], len(self.data_cfg.MD3D_METRICS)), dtype=self.dtype)

        for i in range(output_bboxes_arr.shape[0]):
            # Get 3d bbs after nms on rest of bboxes_arr
            bev_iou = self.get_bev_iou(np.asmatrix(output_bboxes_arr[i, :]))
            iou3d = self.get_iou3d(np.asmatrix(output_bboxes_arr[i, :]), iou_thresh)
                        
            # Candidates entries: x_center, y_center, z_center, l, w, h, theta, volume, surface_area, vol/surf, pid, bev_iou
            candidates = self.get_candidates(bev_iou, iou_thresh)
            
            # TODO: maybe add score later (but how? Max Score, Score_Sum?) -> For now both
#             metrics_single_pc[i, :] = self.compute_metrics(candidates, output_bboxes_arr, iou3d, self.data_cfg.od_iou_threshold, i)
            metrics_single_pc[i, :] = self.compute_metrics(candidates, output_bboxes_arr, iou3d, iou_thresh, i, self.centerpoint_bool)
        
        iou_single_pc = self.get_true_bev_iou(output_bboxes_arr)
        
        if len(iou_single_pc) > 0:
            return np.concatenate([metrics_single_pc, iou_single_pc, np.transpose(np.asmatrix(np.arange(len(iou_single_pc))))], axis=1).astype(self.dtype)
        else:
            return []
    
    
    def get_iou3d(self, bbox, iou_thresh) -> np.ndarray:
        iou3d = bbox_overlaps_3d(
            torch.tensor(bbox[:, :7]) + torch.empty(bbox[:, :7].shape).normal_(mean=0, std=0.00001), 
            torch.tensor(self.proposal_bboxes_single_pc[:, :7]),
            mode='iou',
            coordinate='lidar'
        )

#         return np.asarray(iou3d[iou3d > self.data_cfg.od_iou_threshold])
        return np.asarray(iou3d[iou3d > iou_thresh])
    
    
    def get_bev_iou(self, bbox) -> np.ndarray:
        """
        How nms in Voxelnet, especially Anchor3D-Head works
        
        # Do not perform classical nms but select all boxes that have iou > iou_threshold for further metrics computation
        
        Input:
        
        """
        # Compute BEV version of 3D BBoxes
        mlvl_bboxes_for_nms = xywhr2xyxyr(LiDARInstance3DBoxes(self.proposal_bboxes_single_pc[:, :7]).bev).cuda()
        box_xyxyr = xywhr2xyxyr(LiDARInstance3DBoxes(bbox[:, :7]).bev)
        box_xyxyr += torch.empty(box_xyxyr.size()).normal_(mean=0, std=0.00001)
        
        # Compute IoU of current 3D BBox and all other 3D BBoxes in BEV
        iou = boxes_iou_bev(box_xyxyr.cuda(), mlvl_bboxes_for_nms).cpu()[0]

        return np.asarray(iou)
    
        
    def get_true_bev_iou(self, bbox) -> np.ndarray:
        """
        How nms in Voxelnet, especially Anchor3D-Head works:
        
        Input:
        
        """
        # Compute BEV version of 3D BBoxes
        mlvl_bboxes_for_nms = xywhr2xyxyr(LiDARInstance3DBoxes(self.labels_single_pc).bev).cuda()
        box_xyxyr = xywhr2xyxyr(LiDARInstance3DBoxes(bbox[:, :7]).bev).cuda()
        box_xyxyr += torch.empty(box_xyxyr.size()).normal_(mean=0, std=0.00001).cuda()
        
        # Compute IoU of current 3D BBox and all other 3D BBoxes in BEV
        iou = boxes_iou_bev(box_xyxyr, mlvl_bboxes_for_nms).cpu().numpy()
        if iou.shape[-1] == 0:
            return np.zeros((len(iou), 1))
        if len(iou) > 0:
            return np.amax(iou, axis=1).reshape(iou.shape[0], -1)
        else:
            return np.amax(iou, axis=1)
        
        
    def get_candidates(self, iou, iou_thresh) -> np.ndarray:
        """
        Stores only 3D BBoxes that have iou > self.data_cfg.od_iou_threshold with current 3D BBox
        
        iou_mask: Either bev_iou or iou3d_mask
        """
#         iou_mask = iou > self.data_cfg.od_iou_threshold
        iou_mask = iou > iou_thresh
        
        if self.centerpoint_bool:
            # Initialize candidates
            candidates = np.zeros((len(iou[iou_mask]), 17))

            # Fill candidates 2d array
            candidates[:, :7] = np.asarray(self.proposal_bboxes_single_pc[iou_mask])[:, :7]    # x_center, y_center, z_center, l, w, h, theta
            candidates[:, 7] = candidates[:, 3] * candidates[:, 4] * candidates[:, 5]          # volume
            candidates[:, 8] = 2 * candidates[:, 3] * candidates[:, 4] + 2 * candidates[:, 4] * candidates[:, 5] + 2 * candidates[:, 3] * candidates[:, 5] # surface area
            candidates[:, 9] = candidates[:, 7] / candidates[:, 8]                             # volume/surface area
            candidates[:, 10] = np.asarray(self.proposal_bboxes_single_pc[iou_mask][:, 9])     # pib
            candidates[:, 11] = np.asarray(self.proposal_bboxes_single_pc[iou_mask][:, 10])    # pib_per_total_points
            candidates[:, 12] = np.asarray(self.proposal_bboxes_single_pc[iou_mask][:, 11])    # pib reflectance max
            candidates[:, 13] = np.asarray(self.proposal_bboxes_single_pc[iou_mask][:, 12])    # pib reflectance mean
            candidates[:, 14] = np.asarray(self.proposal_bboxes_single_pc[iou_mask][:, 13])    # pib reflectance std
            candidates[:, 15] = iou[iou_mask]                                                  # iou
            candidates[:, 16] = np.asarray(self.proposal_bboxes_single_pc[iou_mask][:, 7])     # max score
        
        else:
            # Initialize candidates
            candidates = np.zeros((len(iou[iou_mask]), 18))

            # Fill candidates 2d array
            candidates[:, :7] = np.asarray(self.proposal_bboxes_single_pc[iou_mask])[:, :7]    # x_center, y_center, z_center, l, w, h, theta
            candidates[:, 7] = candidates[:, 3] * candidates[:, 4] * candidates[:, 5]          # volume
            candidates[:, 8] = 2 * candidates[:, 3] * candidates[:, 4] + 2 * candidates[:, 4] * candidates[:, 5] + 2 * candidates[:, 3] * candidates[:, 5] # surface area
            candidates[:, 9] = candidates[:, 7] / candidates[:, 8]                             # volume/surface area
            candidates[:, 10] = np.asarray(self.proposal_bboxes_single_pc[iou_mask][:, 14])    # pib
            candidates[:, 11] = np.asarray(self.proposal_bboxes_single_pc[iou_mask][:, 15])    # pib_per_total_points
            candidates[:, 12] = np.asarray(self.proposal_bboxes_single_pc[iou_mask][:, 16])    # pib reflectance max
            candidates[:, 13] = np.asarray(self.proposal_bboxes_single_pc[iou_mask][:, 17])    # pib reflectance mean
            candidates[:, 14] = np.asarray(self.proposal_bboxes_single_pc[iou_mask][:, 18])    # pib reflectance std
            candidates[:, 15] = iou[iou_mask]                                                  # iou
            candidates[:, 16] = np.asarray(self.proposal_bboxes_single_pc[iou_mask][:, 8])     # max score
            candidates[:, 17] = np.asarray(self.proposal_bboxes_single_pc[iou_mask][:, 9])     # sum scores
        
        return candidates
    
    @staticmethod
    @numba.jit(nopython=True)
    def compute_metrics(candidates, bboxes, iou3d, iou_thresh, i, centerpoint_bool) -> np.ndarray:
        bev_iou = ((np.asarray(candidates[:, 15])).flatten())
        bev_iou[np.argmax(bev_iou)] = 0 # set max bev_iou value to 0 (since it is the bev_iou of box with itself)
        # print(f"BEV IoU: {bev_iou}, {bev_iou.shape}")

        if len(bev_iou) > 1:
            bev_iou_min = np.min(bev_iou[bev_iou >= iou_thresh])
            bev_iou_max = np.max(bev_iou)
            bev_iou_mean = np.mean(bev_iou[bev_iou >= iou_thresh])
            bev_iou_std = np.std(bev_iou[bev_iou >= iou_thresh])
        else:
            bev_iou_min, bev_iou_max, bev_iou_mean, bev_iou_std = 0.0, 0.0, 0.0, 0.0
            
        iou3d = ((np.asarray(iou3d)).flatten())
        iou3d[np.argmax(iou3d)] = 0
        
        if len(iou3d) > 1:
            iou3d_min = np.min(iou3d[iou3d >= iou_thresh])
            iou3d_max = np.max(iou3d)
            iou3d_mean = np.mean(iou3d[iou3d >= iou_thresh])
            iou3d_std = np.std(iou3d[iou3d >= iou_thresh])
        else:
            iou3d_min, iou3d_max, iou3d_mean, iou3d_std = 0.0, 0.0, 0.0, 0.0
        
        if centerpoint_bool:
            return np.asarray([
                candidates.shape[0],        # number of candidate boxes

                np.min(candidates[:, 0]),   # x_min
                np.max(candidates[:, 0]),   # x_max
                np.mean(candidates[:, 0]),  # x_mean
                np.std(candidates[:, 0]),   # x_std

                np.min(candidates[:, 1]),   # y_min
                np.max(candidates[:, 1]),   # y_max
                np.mean(candidates[:, 1]),  # y_mean
                np.std(candidates[:, 1]),   # y_std

                np.min(candidates[:, 2]),   # z_min
                np.max(candidates[:, 2]),   # z_max
                np.mean(candidates[:, 2]),  # z_mean
                np.std(candidates[:, 2]),   # z_std

                np.min(candidates[:, 3]),   # l_min
                np.max(candidates[:, 3]),   # l_max
                np.mean(candidates[:, 3]),  # l_mean
                np.std(candidates[:, 3]),   # l_std

                np.min(candidates[:, 4]),   # w_min
                np.max(candidates[:, 4]),   # w_max
                np.mean(candidates[:, 4]),  # w_mean
                np.std(candidates[:, 4]),   # w_std

                np.min(candidates[:, 5]),   # z_min
                np.max(candidates[:, 5]),   # z_max
                np.mean(candidates[:, 5]),  # z_mean
                np.std(candidates[:, 5]),   # z_std

                np.min(candidates[:, 6]),   # theta_min
                np.max(candidates[:, 6]),   # theta_max
                np.mean(candidates[:, 6]),  # theta_mean
                np.std(candidates[:, 6]),   # theta_std

                bboxes[i, 3] * bboxes[i, 4] * bboxes[i, 5], # volume
                np.min(candidates[:, 7]),   # vol_min
                np.max(candidates[:, 7]),   # vol_max
                np.mean(candidates[:, 7]),  # vol_mean
                np.std(candidates[:, 7]),   # vol_std

                (2 * (bboxes[i, 3] * bboxes[i, 4]) + 2 * (bboxes[i, 3] * bboxes[i, 5]) + 2 * (bboxes[i, 4] * bboxes[i, 5])), # surface_area
                np.min(candidates[:, 8]),   # surf_min
                np.max(candidates[:, 8]),   # surf_max
                np.mean(candidates[:, 8]),  # surf_mean
                np.std(candidates[:, 8]),   # surf_std

                (bboxes[i, 3] * bboxes[i, 4] * bboxes[i, 5]) / (2 * (bboxes[i, 3] * bboxes[i, 4]) + 2 * (bboxes[i, 3] * bboxes[i, 5]) + 2 * (bboxes[i, 4] * bboxes[i, 5])),  # volume/surface area
                np.min(candidates[:, 9]),   # vol/surf_min
                np.max(candidates[:, 9]),   # vol/surf_max
                np.mean(candidates[:, 9]),  # vol/surf_mean
                np.std(candidates[:, 9]),   # vol/surf_std

                np.min(candidates[:, 10]),  # pib_min
                np.max(candidates[:, 10]),  # pib_max
                np.mean(candidates[:, 10]), # pib_mean
                np.std(candidates[:, 10]),  # pib_std

                np.min(candidates[:, 11]),  # pib_per_total_points_min
                np.max(candidates[:, 11]),  # pib_per_total_points_max
                np.mean(candidates[:, 11]), # pib_per_total_points_mean
                np.std(candidates[:, 11]),  # pib_per_total_points_std

                np.min(candidates[:, 12]),  # max_reflectance_min
                np.max(candidates[:, 12]),  # max_reflectance_max
                np.mean(candidates[:, 12]), # max_reflectance_mean
                np.std(candidates[:, 12]),  # max_reflectance_std

                np.min(candidates[:, 13]),  # mean_reflectance_min
                np.max(candidates[:, 13]),  # mean_reflectance_max
                np.mean(candidates[:, 13]), # mean_reflectance_mean
                np.std(candidates[:, 13]),  # mean_reflectance_std

                np.min(candidates[:, 14]),  # std_reflectance_min
                np.max(candidates[:, 14]),  # std_reflectance_max
                np.mean(candidates[:, 14]), # std_reflectance_mean
                np.std(candidates[:, 14]),  # std_reflectance_std

                bev_iou_min,                # BEV_IoU_min
                bev_iou_max,                # BEV_IoU_max
                bev_iou_mean,               # BEV_IoU_mean
                bev_iou_std,                # BEV_IoU_std

                iou3d_min,                  # IoU3D_min
                iou3d_max,                  # IoU3D_max
                iou3d_mean,                 # IoU3D_mean
                iou3d_std,                  # IoU3D_std

                np.min(candidates[:, 16]),  # max_score_min
                np.max(candidates[:, 16]),  # max_score_max
                np.mean(candidates[:, 16]), # max_score_mean
                np.std(candidates[:, 16]),  # max_score_std
            ])
        else:
            return np.asarray([
                candidates.shape[0],        # number of candidate boxes

                np.min(candidates[:, 0]),   # x_min
                np.max(candidates[:, 0]),   # x_max
                np.mean(candidates[:, 0]),  # x_mean
                np.std(candidates[:, 0]),   # x_std

                np.min(candidates[:, 1]),   # y_min
                np.max(candidates[:, 1]),   # y_max
                np.mean(candidates[:, 1]),  # y_mean
                np.std(candidates[:, 1]),   # y_std

                np.min(candidates[:, 2]),   # z_min
                np.max(candidates[:, 2]),   # z_max
                np.mean(candidates[:, 2]),  # z_mean
                np.std(candidates[:, 2]),   # z_std

                np.min(candidates[:, 3]),   # l_min
                np.max(candidates[:, 3]),   # l_max
                np.mean(candidates[:, 3]),  # l_mean
                np.std(candidates[:, 3]),   # l_std

                np.min(candidates[:, 4]),   # w_min
                np.max(candidates[:, 4]),   # w_max
                np.mean(candidates[:, 4]),  # w_mean
                np.std(candidates[:, 4]),   # w_std

                np.min(candidates[:, 5]),   # z_min
                np.max(candidates[:, 5]),   # z_max
                np.mean(candidates[:, 5]),  # z_mean
                np.std(candidates[:, 5]),   # z_std

                np.min(candidates[:, 6]),   # theta_min
                np.max(candidates[:, 6]),   # theta_max
                np.mean(candidates[:, 6]),  # theta_mean
                np.std(candidates[:, 6]),   # theta_std

                bboxes[i, 3] * bboxes[i, 4] * bboxes[i, 5], # volume
                np.min(candidates[:, 7]),   # vol_min
                np.max(candidates[:, 7]),   # vol_max
                np.mean(candidates[:, 7]),  # vol_mean
                np.std(candidates[:, 7]),   # vol_std

                (2 * (bboxes[i, 3] * bboxes[i, 4]) + 2 * (bboxes[i, 3] * bboxes[i, 5]) + 2 * (bboxes[i, 4] * bboxes[i, 5])), # surface_area
                np.min(candidates[:, 8]),   # surf_min
                np.max(candidates[:, 8]),   # surf_max
                np.mean(candidates[:, 8]),  # surf_mean
                np.std(candidates[:, 8]),   # surf_std

                (bboxes[i, 3] * bboxes[i, 4] * bboxes[i, 5]) / (2 * (bboxes[i, 3] * bboxes[i, 4]) + 2 * (bboxes[i, 3] * bboxes[i, 5]) + 2 * (bboxes[i, 4] * bboxes[i, 5])),  # volume/surface area
                np.min(candidates[:, 9]),   # vol/surf_min
                np.max(candidates[:, 9]),   # vol/surf_max
                np.mean(candidates[:, 9]),  # vol/surf_mean
                np.std(candidates[:, 9]),   # vol/surf_std

                np.min(candidates[:, 10]),  # pib_min
                np.max(candidates[:, 10]),  # pib_max
                np.mean(candidates[:, 10]), # pib_mean
                np.std(candidates[:, 10]),  # pib_std

                np.min(candidates[:, 11]),  # pib_per_total_points_min
                np.max(candidates[:, 11]),  # pib_per_total_points_max
                np.mean(candidates[:, 11]), # pib_per_total_points_mean
                np.std(candidates[:, 11]),  # pib_per_total_points_std

                np.min(candidates[:, 12]),  # max_reflectance_min
                np.max(candidates[:, 12]),  # max_reflectance_max
                np.mean(candidates[:, 12]), # max_reflectance_mean
                np.std(candidates[:, 12]),  # max_reflectance_std

                np.min(candidates[:, 13]),  # mean_reflectance_min
                np.max(candidates[:, 13]),  # mean_reflectance_max
                np.mean(candidates[:, 13]), # mean_reflectance_mean
                np.std(candidates[:, 13]),  # mean_reflectance_std

                np.min(candidates[:, 14]),  # std_reflectance_min
                np.max(candidates[:, 14]),  # std_reflectance_max
                np.mean(candidates[:, 14]), # std_reflectance_mean
                np.std(candidates[:, 14]),  # std_reflectance_std

                bev_iou_min,                # BEV_IoU_min
                bev_iou_max,                # BEV_IoU_max
                bev_iou_mean,               # BEV_IoU_mean
                bev_iou_std,                # BEV_IoU_std

                iou3d_min,                  # IoU3D_min
                iou3d_max,                  # IoU3D_max
                iou3d_mean,                 # IoU3D_mean
                iou3d_std,                  # IoU3D_std

                np.min(candidates[:, 16]),  # max_score_min
                np.max(candidates[:, 16]),  # max_score_max
                np.mean(candidates[:, 16]), # max_score_mean
                np.std(candidates[:, 16]),  # max_score_std

                np.min(candidates[:, 17]),  # sum_scores_min
                np.max(candidates[:, 17]),  # sum_scores_max
                np.mean(candidates[:, 17]), # sum_scores_mean
                np.std(candidates[:, 17]),  # sum_scores_std
            ])
    