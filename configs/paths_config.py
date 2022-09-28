from os import path as osp

ROOT_DIR = "/home/jkezmann/mmdet_venv"
MMDET3D_DIR = osp.join(ROOT_DIR, "mmdetection3d")
MMDET3D_CONFIGS_DIR = osp.join(MMDET3D_DIR, "configs")
MMDET3D_CKPTS_DIR = osp.join(MMDET3D_DIR, "checkpoints")
MMDET3D_OUT_DIR = osp.join(MMDET3D_DIR, "outputs")

KITTI_BASE_DIR = osp.join(MMDET3D_DIR, "data/kitti")
KITTI_TRAIN_DIR = osp.join(KITTI_BASE_DIR, "training")
KITTI_TEST_DIR = osp.join(KITTI_BASE_DIR, "testing")
KITTI_TRAIN_LABELS_DIR = osp.join(KITTI_TRAIN_DIR, "label_2")
KITTI_TRAIN_VELO_DIR = osp.join(KITTI_TRAIN_DIR, "velodyne")
KITTI_TEST_VELO_DIR = osp.join(KITTI_TEST_DIR, "velodyne")
KITTI_TRAIN_VELO_RED_DIR = osp.join(KITTI_TRAIN_DIR, "velodyne_reduced")
KITTI_TEST_VELO_RED_DIR = osp.join(KITTI_TEST_DIR, "velodyne_reduced")
KITTI_TRAIN_IMAGE_SETS_DIR = osp.join(KITTI_BASE_DIR, "ImageSets")

infos_train_path = osp.join(KITTI_BASE_DIR, "kitti_infos_train.pkl")
img_sets_train_path = osp.join(KITTI_TRAIN_IMAGE_SETS_DIR, "train.txt")
infos_trainval_path = osp.join(KITTI_BASE_DIR, "kitti_infos_trainval.pkl")
img_sets_trainval_path = osp.join(KITTI_TRAIN_IMAGE_SETS_DIR, "trainval.txt")

NUSCENES_BASE_DIR = osp.join(MMDET3D_DIR, "data/nuscenes")
NUSCENES_MAPS_DIR = osp.join(NUSCENES_BASE_DIR, "maps")
NUSCENES_SAMPLES_DIR = osp.join(NUSCENES_BASE_DIR, "samples")
NUSCENES_SWEEPS_DIR = osp.join(NUSCENES_BASE_DIR, "sweeps")
NUSCENES_TRAINVAL_DIR = osp.join(NUSCENES_BASE_DIR, "v1.0-trainval")
NUSCENES_TEST_DIR = osp.join(NUSCENES_BASE_DIR, "v1.0-test")

MD3D_DIR = osp.join(ROOT_DIR, "MetaDetect3D")
MD3D_REPO_DIR = osp.join(MD3D_DIR, "metadetect3d")
MD3D_CONFIGS_DIR = osp.join(MD3D_REPO_DIR, "configs")
MD3D_OUT_DIR = osp.join(MD3D_DIR, "outputs")

base_aptiv_dir = osp.join(MMDET3D_DIR, "data/aptiv")
aptiv_lidar_dir = osp.join(base_aptiv_dir, "lidar")
gt_db_dir = osp.join(base_aptiv_dir, "aptiv_gt_database")
sdb_infos_trainval_path = osp.join(base_aptiv_dir, "annotation.pkl")
sdb_infos_train_path = osp.join(base_aptiv_dir, "annotation_train.pkl")
sdb_infos_val_path = osp.join(base_aptiv_dir, "annotation_val.pkl")
sdb_infos_test_path = osp.join(base_aptiv_dir, "annotation_test.pkl")
sdb_infos_valtest_path = osp.join(base_aptiv_dir, "annotation_valtest.pkl")
sdb_infos_shorttest_path = osp.join(base_aptiv_dir, "annotation_shorttest.pkl")

aptiv_dbinfos_trainvaltest_path = osp.join(base_aptiv_dir, "aptiv_dbinfos_trainvaltest.pkl")
aptiv_dbinfos_train_path = osp.join(base_aptiv_dir, "aptiv_dbinfos_train.pkl")
aptiv_dbinfos_val_path = osp.join(base_aptiv_dir, "aptiv_dbinfos_val.pkl")
aptiv_dbinfos_test_path = osp.join(base_aptiv_dir, "aptiv_dbinfos_test.pkl")