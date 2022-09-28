import os
import pandas as pd
from os import path as osp

try:
    from configs import paths_config as pc
except ImportError:
    from .. import paths_config as pc
    
#################################################################
# Below Must be set
#################################################################
model_name = "sdb4cls_centerpoint_0075voxel_rx96ry48"

model_cfg_path = osp.join(pc.MMDET3D_OUT_DIR, f"centerpoint/{model_name}/{model_name}.py")
model_ckpt_path = osp.join(pc.MMDET3D_OUT_DIR, f"centerpoint/{model_name}/epoch_35.pth")

#################################################################
# Below can but should not be set
#################################################################
dataset_name = "aptiv"

infos_train_path = pc.sdb_infos_train_path
infos_val_path = pc.sdb_infos_val_path
infos_test_path = pc.sdb_infos_test_path

infos_val = pd.read_pickle(infos_val_path)
infos_test = pd.read_pickle(infos_test_path)
img_sets_val_path = [info["lidar_points"]["lidar_path"] for info in infos_val]
img_sets_test_path = [info["lidar_points"]["lidar_path"] for info in infos_test]

class_to_num = {'car_or_van_or_suv_or_pickup': 0, 'truck_or_bus': 1, 'pedestrian': 2, 'motorcycle_or_bicycle': 3}

num_to_class = {0: 'car_or_van_or_suv_or_pickup', 1: 'truck_or_bus', 2: 'pedestrian', 3: 'motorcycle_or_bicycle'}

meta_metrics_score_thresholds = [0.1, 0.3]
meta_metrics_iou_thresholds = [0.5]

seed = 5

# field names
# pib -> points in box
PRED_PROPOSAL_FIELDS = [
    "file_path",
    "x_center", "y_center", "z_center",
    "length", "width", "height",
    "theta",
    "max_score", "category_idx",
    "pib", "pib_per_total_points", "max_reflectance", "mean_reflectance", "std_reflectance",
    "dataset_box_id"
]

PRED_OUTPUT_FIELDS = [
    "file_path",
    "x_center", "y_center", "z_center",
    "length", "width", "height",
    "theta",
    "score", "category_idx",
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
]

LABEL_METRICS = ["True_BEV_IoU", "dataset_box_id"]

