import os
from os import path as osp

try:
    from configs import paths_config as pc
except ImportError:
    from .. import paths_config as pc
    
#################################################################
# Below Must be set
#################################################################
#model_name = "hv_pointpillars_fpn_sbn-all_fp16_2x8_2x_nus-3d"
model_name = "hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d"

model_cfg_path = osp.join(pc.MMDET3D_OUT_DIR, f"pointpillars/{model_name}/{model_name}.py")
#model_ckpt_path = osp.join(pc.MMDET3D_OUT_DIR, f"pointpillars/{model_name}/epoch_24.pth")
model_ckpt_path = osp.join(pc.MMDET3D_CKPTS_DIR, "pointpillars/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20210826_104936-fca299c1.pth")

#################################################################
# Below can but should not be set
#################################################################
dataset_name = "nuscenes"

infos_train_path = osp.join(pc.NUSCENES_BASE_DIR, "nuscenes_infos_train.pkl")
infos_val_path = osp.join(pc.NUSCENES_BASE_DIR, "nuscenes_infos_val.pkl")
infos_test_path = osp.join(pc.NUSCENES_BASE_DIR, "nuscenes_infos_test.pkl")

img_sets_val_path = [
    "n015-2018-08-02-17-16-37+0800", "n015-2018-07-18-11-41-49+0800", "n015-2018-07-24-10-42-41+0800",
    "n008-2018-08-01-15-16-36-0400", "n008-2018-08-22-15-53-49-0400", "n008-2018-08-30-10-33-52-0400",
    "n008-2018-09-18-15-12-01-0400", "n015-2018-10-08-15-36-50+0800", "n015-2018-11-21-19-21-35+0800"
]

img_sets_test_path = [
    "n015-2018-07-16-11-49-16+0800", "n015-2018-07-11-11-54-16+0800", "n015-2018-09-25-13-17-43+0800",
    "n008-2018-08-28-16-43-51-0400", "n008-2018-08-31-11-37-23-0400", "n008-2018-09-18-14-35-12-0400",
    "n008-2018-08-30-15-31-50-0400", "n015-2018-10-02-10-50-40+0800", "n015-2018-10-08-15-44-23+0800"
]

class_to_num = {"car": 0, "truck": 1, "trailer": 2, "bus": 3, "construction_vehicle": 4, "bicycle": 5,
                "motorcycle": 6, "pedestrian": 7, "traffic_cone": 8, "barrier": 9}
num_to_class = {0: 'car', 1: 'truck', 2: 'trailer', 3: 'bus', 4: 'construction_vehicle', 5: 'bicycle',
                6: 'motorcycle', 7: 'pedestrian', 8: 'traffic_cone', 9: 'barrier'}

meta_metrics_score_thresholds = [0.1, 0.3]
meta_metrics_iou_thresholds = [0.2, 0.3, 0.4, 0.5]

seed = 5

# field names
# pib -> points in box
PRED_PROPOSAL_FIELDS = [
    "file_path",
    "x_center", "y_center", "z_center",
    "length", "width", "height",
    "theta",
    "dir_score", "max_score", "score_sum", "score_0", "score_1", "score_2", "score_3", "score_4", "score_5", "score_6", "score_7", "score_8", "score_9", "category_idx",
    "pib", "pib_per_total_points", "max_reflectance", "mean_reflectance", "std_reflectance",
    "dataset_box_id"
]

PRED_OUTPUT_FIELDS = [
    "file_path",
    "x_center", "y_center", "z_center",
    "length", "width", "height",
    "theta",
    "dir_score", "score", "category_idx",
    "pib", "pib_per_total_points", "max_reflectance", "mean_reflectance", "std_reflectance",
    "dataset_box_id"
]

# OUTPUT_METRICS = PRED_FIELDS[:8] + PRED_FIELDS[9:-1]

GT_FIELDS = [
    "file_path",
    "x_center", "y_center", "z_center",
    "length", "width", "height",
    "theta",
    "category_idx",
    "pib",
    "dataset_box_id"
]

MD3D_METRICS = [
    "Number of Candidate Boxes",
    "x_center_min", "x_center_max", "x_center_mean", "x_center_std",
    "y_center_min", "y_center_max", "y_center_mean", "y_center_std",
    "z_center_min", "z_center_max", "z_center_mean", "z_center_std",
    "l_min", "l_max", "l_mean", "l_std",
    "w_min", "w_max", "w_mean", "w_std",
    "h_min", "h_max", "h_mean", "h_std",
    "theta_min", "theta_max", "theta_mean", "theta_std",
    "volume", "volumne_min", "volume_max", "volume_mean", "volume_std",
    "surface_area", "surface_area_min", "surface_area_max", "surface_area_mean", "surface_area_std",
    "vol/surface_area", "vol/surface_area_min", "vol/surface_area_max", "vol/surface_area_mean", "vol/surface_area_std",
    #"dir_score_min", "dir_score_max", "dir_score_mean", "dir_score_std",
    "pib_min", "pib_max", "pib_mean", "pib_std",
    "pib_per_total_points_min", "pib_per_total_points_max", "pib_per_total_points_mean", "pib_per_total_points_std",
    "max_reflectance_min", "max_reflectance_max", "max_reflectance_mean", "max_reflectance_std",
    "mean_reflectance_min", "mean_reflectance_max", "mean_reflectance_mean", "mean_reflectance_std",
    "std_reflectance_min", "std_reflectance_max", "std_reflectance_mean", "std_reflectance_std",
    "BEV_IoU_min", "BEV_IoU_max", "BEV_IoU_mean", "BEV_IoU_std",
    "IoU3D_min", "IoU3D_max", "IoU3D_mean", "IoU3D_std",
    "max_score_min", "max_score_max", "max_score_mean", "max_score_std",
    "sum_scores_min", "sum_scores_max", "sum_scores_mean", "sum_scores_std",
]

LABEL_METRICS = ["True_BEV_IoU", "dataset_box_id"]

