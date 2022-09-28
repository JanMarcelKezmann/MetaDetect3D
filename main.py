import os
import sys
#SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
#sys.path.append(os.path.dirname(SCRIPT_DIR))
import time
import shutil
import inspect
import warnings
import argparse
import importlib

from os import path as osp

from mmcv import Config
from mmdet3d.utils import get_root_logger

import configs
from src import AptivDataCrawler, NuScenesDataCrawler, KITTIDataCrawler, MetaDetect3DMetrics, Trainer


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser(description='Load or Preprocess OD and MetaDetect data and train and optimize MetaDetect3D models.')
    #parser.add_argument('--experiment-name', type=str, help="Name of experiments in configs (for details see ReadMe.", required=True)
    
    parser.add_argument('--save-dir', type=str, help="Local directory in 'outputs' where files will be saved to. Directory will equal experiment name.", required=True)
    parser.add_argument('--config-dir', type=str, default="configs/", help="Directory where all configs are stored")
    parser.add_argument('--path-cfg', type=str, default="paths_config.py", help="Directory where all configs are stored")
    parser.add_argument('--model-cfg', type=str, default="meta_models_configs/best_models.py", help="Directory where all configs are stored")
    parser.add_argument('--data-cfg', type=str, help="Directory where all configs are stored", required=True)
    
    parser.add_argument("--crawl_od_data", type=str2bool, nargs='?', const=True, help="Whether to crawl prediction and GT bounding boxes from 3D OD, necessary if not existing already.", required=True)
    parser.add_argument('--compute_metrics', type=str2bool, nargs='?', const=True, help="Whether to compute data for MetaDetect3D, necessary if not existing already.", required=True)
    parser.add_argument('--only_eval', type=str2bool, nargs='?', default="no", const=True, help="If set to false only training and evaluation is done and no parameter search.")
    
    parser.add_argument('--gpu-id', type=int, default=0, help="Id of GPU to use")
    parser.add_argument('--n_jobs', type=int, default=30, help="Number of CPUs for training of meta classfiers and regressors.")
    
    args = parser.parse_args()
    
    return args

def get_log_and_save_configs(args, logger):
    path_cfg = Config.fromfile(osp.join(args.config_dir, args.path_cfg))
    data_cfg = Config.fromfile(osp.join(args.config_dir, args.data_cfg))
    model_cfg = Config.fromfile(osp.join(args.config_dir, args.model_cfg))
    
    # log configs and remove modules from configs
    logger.info("\nPath Config:")
    for k, v in list(path_cfg.items()):
        if inspect.ismodule(v):
            path_cfg.pop(k)
        else:
            logger.info(f'{k}: {v}')
            
    logger.info('\nData Config:')
    for k, v in list(data_cfg.items()):
        if inspect.ismodule(v):
            data_cfg.pop(k)
    #    else:
    #        logger.info(f'{k}: {v}')
    
    logger.info('\nModel Config:')
    for k, v in list(model_cfg.items()):
        if inspect.ismodule(v):
            model_cfg.pop(k)
        else:
            logger.info(f'{k}: {v}')
            
    # Add GT and Pred Directories for OD data to data_cfg
    data_cfg.pred_dir = osp.join(args.save_dir, "prediction/")
    data_cfg.gt_dir = osp.join(args.save_dir, "ground_truth/")
    
    # Copy and save configs in save_dir
    os.makedirs(osp.join(args.save_dir, "configs/"), exist_ok=True)
    shutil.copy(osp.join(args.config_dir, args.path_cfg), osp.join(args.save_dir, "configs/", args.path_cfg.split("/")[-1]))
    shutil.copy(osp.join(args.config_dir, args.data_cfg), osp.join(args.save_dir, "configs/", args.data_cfg.split("/")[-1]))
    shutil.copy(osp.join(args.config_dir, args.model_cfg), osp.join(args.save_dir, "configs/", args.model_cfg.split("/")[-1]))
    
    os.makedirs(data_cfg.pred_dir, exist_ok=True)
    os.makedirs(data_cfg.gt_dir, exist_ok=True)
    
    # Save OD configs in save_dir
    shutil.copy(data_cfg.model_cfg_path, osp.join(args.save_dir, "configs/", data_cfg.model_cfg_path.split("/")[-1]))
    
    return path_cfg, data_cfg, model_cfg

def main():
    args = parse_args()
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    else:
        print(f"Path {args.save_dir} already exists, i.e., existing files can be overwritten")
        
    # Init logger
    timestamp = time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())
    log_file = osp.join(args.save_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, name=args.save_dir.split("/")[-1])
    
    # Init configs and save them
    path_cfg, data_cfg, model_cfg = get_log_and_save_configs(args, logger)
    
    # set random seed
    logger.info(f'Random seed is set to {data_cfg.get("seed", 0)}')
    logger.info(f'Dataset: {data_cfg.get("dataset_name", 0)}')
    
    device = f'cuda:{args.gpu_id}'
    
    if "kitti" == data_cfg.dataset_name.lower():
        if args.crawl_od_data:
            logger.info(f'Crawl KITTI data.')
    
            kdc = KITTIDataCrawler(
                data_cfg = data_cfg,
                models_cfg = model_cfg,
                path_cfg = path_cfg,
                device = device
            )
            
            kdc.crawl_pred()
            kdc.save_pred_proposal_df()
            kdc.save_pred_output_df()

            kdc.crawl_gt()
            kdc.save_gt_df()
    elif "nuscenes" == data_cfg.dataset_name.lower():
        if args.crawl_od_data:
            logger.info(f'Crawl NuScenes data.')
    
            ndc = NuScenesDataCrawler(
                data_cfg = data_cfg,
                models_cfg = model_cfg,
                path_cfg = path_cfg,
                device = device
            )
            
            ndc.crawl_pred()
            ndc.save_pred_proposal_df()
            ndc.save_pred_output_df()

            ndc.crawl_gt()
            ndc.save_gt_df()
    elif "aptiv" == data_cfg.dataset_name.lower():
        if args.crawl_od_data:
            logger.info(f'Crawl NuScenes data.')
    
            adc = AptivDataCrawler(
                data_cfg = data_cfg,
                models_cfg = model_cfg,
                path_cfg = path_cfg,
                device = device
            )
            
            adc.crawl_pred()
            adc.save_pred_proposal_df()
            adc.save_pred_output_df()

            adc.crawl_gt()
            adc.save_gt_df()
    
    if args.compute_metrics:
        logger.info(f'Compute Meta Metrics.')
        mdm = MetaDetect3DMetrics(
            path_cfg = path_cfg,
            data_cfg = data_cfg, 
            models_cfg = model_cfg,
        )

        mdm.get_and_save_metrics()

    t = Trainer(
        path_cfg = path_cfg,
        data_cfg = data_cfg,
        models_cfg = model_cfg,
        n_jobs = args.n_jobs,
        save_dir = args.save_dir
    )
    
    if not args.only_eval:
        logger.info(f'Start Parameter Search.')
        t.parameter_search()
        #t.parameter_search_best_k()
    
    logger.info(f'Start Training and Evaluation Pipeline.')
    t.train_eval()
    

if __name__ == '__main__':
    main()