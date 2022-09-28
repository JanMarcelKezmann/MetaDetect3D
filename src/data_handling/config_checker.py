import sys
import traceback

from os import path as osp


class ConfigChecker:
    def __init__(self, path_cfg = None, data_cfg = None, models_cfg = None):
        self.path_cfg = path_cfg
        self.data_cfg = data_cfg
        self.models_cfg = models_cfg
            
        if path_cfg is not None:
            self.check_path_config()
        if self.data_cfg is not None:
            self.check_data_config()
        if self.models_cfg is not None:
            self.check_model_config()
            
    def check_path_config(self):
        try:
            assert isinstance(self.path_cfg.ROOT_DIR, str)
            assert self.path_cfg.ROOT_DIR != ""
            assert osp.exists(self.path_cfg.ROOT_DIR)
            
            assert isinstance(self.path_cfg.MMDET3D_DIR, str)
            assert self.path_cfg.MMDET3D_DIR != ""
            assert osp.exists(self.path_cfg.MMDET3D_DIR)
            
            assert isinstance(self.path_cfg.MMDET3D_CONFIGS_DIR, str)
            assert self.path_cfg.MMDET3D_CONFIGS_DIR != ""
            assert osp.exists(self.path_cfg.MMDET3D_CONFIGS_DIR)
            
            assert isinstance(self.path_cfg.MMDET3D_CKPTS_DIR, str)
            assert self.path_cfg.MMDET3D_CKPTS_DIR != ""
            assert osp.exists(self.path_cfg.MMDET3D_CKPTS_DIR)
            
            assert isinstance(self.path_cfg.KITTI_BASE_DIR, str)
            assert self.path_cfg.KITTI_BASE_DIR != ""
            assert osp.exists(self.path_cfg.KITTI_BASE_DIR)
            
            assert isinstance(self.path_cfg.MD3D_DIR, str)
            assert self.path_cfg.MD3D_DIR != ""
            assert osp.exists(self.path_cfg.MD3D_DIR)
            
            assert isinstance(self.path_cfg.MD3D_CONFIGS_DIR, str)
            assert self.path_cfg.MD3D_CONFIGS_DIR != ""
            assert osp.exists(self.path_cfg.MD3D_CONFIGS_DIR)    
            
        except AssertionError:
            _, _, tb = sys.exc_info()
            traceback.print_tb(tb)  # Fixed format
            tb_info = traceback.extract_tb(tb)
            filename, line, func, text = tb_info[-1]

            print('An error occurred on line {} in statement {}'.format(line, text))
            exit(1)
        
    def check_data_config(self):
        try:
            #assert isinstance(self.data_cfg.reduced_velodyne, bool)
            
            #assert isinstance(self.data_cfg.split, str)
            #assert self.data_cfg.split in ["_train_val_test", "_train_val"]
            
            assert isinstance(self.data_cfg.model_cfg_path, str)
            assert self.data_cfg.model_cfg_path != ""
            assert osp.exists(self.data_cfg.model_cfg_path)
            
            assert isinstance(self.data_cfg.model_ckpt_path, str)
            assert self.data_cfg.model_ckpt_path != ""
            assert osp.exists(self.data_cfg.model_ckpt_path)
            
            assert isinstance(self.data_cfg.class_to_num, dict)
            assert isinstance(self.data_cfg.num_to_class, dict)
            assert len(self.data_cfg.class_to_num) == len(self.data_cfg.num_to_class)
            assert sorted(list(self.data_cfg.class_to_num.keys())) == sorted(list(self.data_cfg.num_to_class.values()))
            assert sorted(list(self.data_cfg.class_to_num.values())) == sorted(list(self.data_cfg.num_to_class.keys()))
            
#             assert isinstance(self.data_cfg.od_score_threshold, float)
#             assert self.data_cfg.od_score_threshold >= 0.0
#             assert self.data_cfg.od_score_threshold <= 1.0
            
#             assert isinstance(self.data_cfg.od_iou_threshold, float)
#             assert self.data_cfg.od_iou_threshold >= 0.0
#             assert self.data_cfg.od_iou_threshold <= 1.0
            
#             assert isinstance(self.data_cfg.md_iou_threshold, float)
#             assert self.data_cfg.md_iou_threshold >= 0.0
#             assert self.data_cfg.md_iou_threshold <= 1.0
            
            assert isinstance(self.data_cfg.PRED_PROPOSAL_FIELDS, list)
            assert isinstance(self.data_cfg.PRED_OUTPUT_FIELDS, list)
            assert isinstance(self.data_cfg.GT_FIELDS, list)
            assert isinstance(self.data_cfg.MD3D_METRICS, list)
            
        except AssertionError:
            _, _, tb = sys.exc_info()
            traceback.print_tb(tb)  # Fixed format
            tb_info = traceback.extract_tb(tb)
            filename, line, func, text = tb_info[-1]

            print('An error occurred on line {} in statement {}'.format(line, text))
            exit(1)
            
    def check_model_config(self):
        try:
            pass
            
        except AssertionError:
            _, _, tb = sys.exc_info()
            traceback.print_tb(tb)  # Fixed format
            tb_info = traceback.extract_tb(tb)
            filename, line, func, text = tb_info[-1]

            print('An error occurred on line {} in statement {}'.format(line, text))
            exit(1)