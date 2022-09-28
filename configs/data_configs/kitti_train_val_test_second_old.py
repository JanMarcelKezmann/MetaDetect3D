import os
from os import path as osp

try:
    from configs import paths_config as pc
except ImportError:
    from .. import paths_config as pc
    
model_name = "hv_second_secfpn_6x8_80e_kitti-3d-3class"
split = "_train_val_test"
reduced_velodyne = False
velodyne_suffix = "_reduced" if reduced_velodyne else ""

kitti_second_data_base_dir = osp.join(pc.ROOT_DIR, f"MetaDetect3D/outputs/experiment3/")
pred_dir = osp.join(kitti_second_data_base_dir, "prediction")
gt_dir = osp.join(kitti_second_data_base_dir, "ground_truth")

model_cfg_path = osp.join(pc.MMDET3D_OUT_DIR, f"second/{model_name}{split}{velodyne_suffix}/{model_name}{split}{velodyne_suffix}.py")
model_ckpt_path = osp.join(pc.MMDET3D_OUT_DIR, f"second/{model_name}{split}{velodyne_suffix}/epoch_32.pth")

infos_val_path = osp.join(pc.KITTI_BASE_DIR, "kitti_infos_val.pkl")
infos_new_val_path = osp.join(pc.KITTI_BASE_DIR, "kitti_infos_new_val.pkl")
infos_new_test_path = osp.join(pc.KITTI_BASE_DIR, "kitti_infos_new_test.pkl")

img_sets_new_val_path = osp.join(pc.KITTI_TRAIN_IMAGE_SETS_DIR, "new_val.txt")
img_sets_new_test_path = osp.join(pc.KITTI_TRAIN_IMAGE_SETS_DIR, "new_test.txt")


meta_model = "linear_regression"

class_to_num = {"Pedestrian": 0, "Cyclist": 1, "Car": 2}
num_to_class = {0: "Pedestrian", 1: "Cyclist", 2: "Car"}

od_score_threshold = 0.1 # Object Detection Score Threshold
od_iou_threshold = 0.45   # Object Detection IoU Threshold
md_iou_threshold = 0.45   # MetaDetect IoU Threshold 

seed = 5


#SCORE_THRESHOLDS_META_DETECT = [0.01, 0.1, 0.3, 0.5]

# field names
# pib -> points in box
PRED_PROPOSAL_FIELDS = [
    "file_path",
    "x_center", "y_center", "z_center",
    "length", "width", "height",
    "theta",
    "dir_score", "max_score", "score_sum", "score_0", "score_1", "score_2", "category_idx",
    "pib",
    "dataset_box_id"
]

PRED_OUTPUT_FIELDS = [
    "file_path",
    "x_center", "y_center", "z_center",
    "length", "width", "height",
    "theta",
    "dir_score", "score", "category_idx",
    "pib",
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
    "BEV_IoU_min", "BEV_IoU_max", "BEV_IoU_mean", "BEV_IoU_std",
    "IoU3D_min", "IoU3D_max", "IoU3D_mean", "IoU3D_std",
    "max_score_min", "max_score_max", "max_score_mean", "max_score_std",
    "sum_scores_min", "sum_scores_max", "sum_scores_mean", "sum_scores_std",
]

LABEL_METRICS = ["True_BEV_IoU", "dataset_box_id"]

