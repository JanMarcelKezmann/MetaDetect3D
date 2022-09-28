import torch

import numpy as np

from copy import deepcopy

from mmcv.parallel import collate, scatter

from mmdet3d.core.bbox import get_box_type
from mmdet3d.datasets.pipelines import Compose
from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes

from .box_processing import *


def kitti_inference_detector(model, pcd):
    """Inference point cloud with the detector.
    Args:
        model (nn.Module): The loaded detector.
        pcd (str): Point cloud files.
    Returns:
        tuple: Predicted results and data from pipeline.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    if not isinstance(pcd, str):
        cfg = cfg.copy()
        # set loading pipeline type
        cfg.data.test.pipeline[0].type = 'LoadPointsFromDict'

    # build the data pipeline
    test_pipeline = deepcopy(cfg.data.test.pipeline)
    test_pipeline = Compose(test_pipeline)

    box_type_3d, box_mode_3d = get_box_type(cfg.data.test.box_type_3d)

    # load from point clouds file
    data = dict(
        pts_filename=pcd,
        box_type_3d=box_type_3d,
        box_mode_3d=box_mode_3d,
        # for ScanNet demo we need axis_align_matrix
        ann_info=dict(axis_align_matrix=np.eye(4)),
        sweeps=[],
        # set timestamp = 0
        timestamp=[0],
        img_fields=[],
        bbox3d_fields=[],
        pts_mask_fields=[],
        pts_seg_fields=[],
        bbox_fields=[],
        mask_fields=[],
        seg_fields=[]
    )

    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device.index])[0]
    else:
        # this is a workaround to avoid the bug of MMDataParallel
        data['img_metas'] = data['img_metas'][0].data
        data['points'] = data['points'][0].data

    # forward the model
    with torch.no_grad():
        if "voxelnet" in str(model.__class__).lower():
            x = model.extract_feat(*data["points"], *data["img_metas"])
            outs = model.bbox_head(x)
            bbox_output, bbox_proposals_list = model.bbox_head.get_meta_bboxes(*outs, *data["img_metas"], rescale=False)
            
            bbox_proposals = [custom_bbox3d2result_proposals(bboxes, scores[:, :3], dir_scores) for bboxes, scores, dir_scores in bbox_proposals_list]
            bbox_output = [custom_bbox3d2result_output(bboxes, scores, labels, dir_scores) for bboxes, scores, labels, dir_scores in bbox_output]
        
        elif "centerpoint" in str(model.__class__).lower():
            img_feats, pts_feats = model.extract_feat(*data["points"], img=None, img_metas=data["img_metas"])
            bbox_list = [dict() for i in range(len(data["img_metas"]))]
            bbox_output, bbox_proposals_list = model.simple_meta_test_pts(pts_feats, *data["img_metas"], rescale=False)
    
            bbox_proposals = [custom_bbox3d2result_output(bboxes, scores, labels, None) for bboxes, scores, labels in bbox_proposals_list]
            bbox_output = [custom_bbox3d2result_output(bboxes, scores, labels, None) for bboxes, scores, labels in bbox_output]

    return bbox_output, bbox_proposals, data


def nuscenes_inference_detector(model, pcd):
    """Inference point cloud with the detector.
    Args:
        model (nn.Module): The loaded detector.
        pcd (str): Point cloud files.
    Returns:
        tuple: Predicted results and data from pipeline.
    """
    cfg = model.cfg
    
    #{'nms_thr': 0.01,'score_thr': 0.1,'nms_pre': 100,'max_num': 50}
    
    cfg["model"]["pts_bbox_head"]["test_cfg"]['nms_thr'] = 0.01
    cfg["model"]["test_cfg"]['nms_thr'] = 0.01
    cfg["model"]["pts_bbox_head"]["test_cfg"]['score_thr'] = 0.1
    cfg["model"]["test_cfg"]['score_thr'] = 0.1
    device = next(model.parameters()).device  # model device

    if not isinstance(pcd, str):
        cfg = cfg.copy()
        # set loading pipeline type
        cfg.data.test.pipeline[0].type = 'LoadPointsFromDict'
            
    # add reflectance/intensity to data
    if "mvxfasterrcnn" in str(model.__class__).lower():
        cfg.data.test.pipeline[1]["use_dim"] = [0, 1, 2, 3, 4]
    elif "centerpoint" in str(model.__class__).lower():
        cfg.data.test.pipeline[1] = {
            'type': 'LoadPointsFromMultiSweeps',
            'sweeps_num': 9,
            'file_client_args': {'backend': 'disk'},
            'use_dim': [0, 1, 2, 3, 4]
            }
    
    
    # build the data pipeline
    test_pipeline = deepcopy(cfg.data.test.pipeline)
    test_pipeline = Compose(test_pipeline)

    box_type_3d, box_mode_3d = get_box_type(cfg.data.test.box_type_3d)

    # load from point clouds file
    data = dict(
        pts_filename=pcd,
        box_type_3d=box_type_3d,
        box_mode_3d=box_mode_3d,
        # for ScanNet demo we need axis_align_matrix
        ann_info=dict(axis_align_matrix=np.eye(4)),
        sweeps=[],
        # set timestamp = 0
        timestamp=[0],
        img_fields=[],
        bbox3d_fields=[],
        pts_mask_fields=[],
        pts_seg_fields=[],
        bbox_fields=[],
        mask_fields=[],
        seg_fields=[]
    )
    
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)

    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device.index])[0]
        # store reflectance/intensities
        intensities = data["points"][0][0].cpu().numpy()[:, 3]
        # set data properly
        if "mvxfasterrcnn" in str(model.__class__).lower():
            data["points"] = [[torch.cat((data["points"][0][0][:, :3], data["points"][0][0][:, 4:]), 1).cuda()]]
    else:
        # this is a workaround to avoid the bug of MMDataParallel
        data['img_metas'] = data['img_metas'][0].data
        data['points'] = data['points'][0].data

    with torch.no_grad():
        if "mvxfasterrcnn" in str(model.__class__).lower():
            img_feats, pts_feats = model.extract_feat(*data["points"], img_metas=data["img_metas"], img=None)
            bbox_list = [dict() for i in range(len(*data["img_metas"]))]

            outs = model.pts_bbox_head(pts_feats)
            bbox_output, bbox_proposals_list = model.pts_bbox_head.get_meta_bboxes(*outs, *data["img_metas"], rescale=False)

            bbox_proposals = [custom_bbox3d2result_proposals(LiDARInstance3DBoxes(bboxes.tensor[:, :7]), scores[:, :10], dir_scores) for bboxes, scores, dir_scores in bbox_proposals_list]
            bbox_output = [custom_bbox3d2result_output(LiDARInstance3DBoxes(bboxes.tensor[:, :7]), scores, labels, dir_scores) for bboxes, scores, labels, dir_scores in bbox_output]

            scores_bool = np.max(bbox_proposals[0]["scores_3d"].numpy(), axis=1) > 0.1

            bbox_proposals = [
                {
                    "boxes_3d": bbox_proposals[0]["boxes_3d"][scores_bool],
                    "scores_3d": bbox_proposals[0]["scores_3d"][scores_bool],
                    "dir_scores_3d": bbox_proposals[0]["dir_scores_3d"][scores_bool]
                }
            ]
        elif "centerpoint" in str(model.__class__).lower():
            img_feats, pts_feats = model.extract_feat(*data["points"], img=None, img_metas=data["img_metas"])
            bbox_list = [dict() for i in range(len(data["img_metas"]))]

            bbox_output, bbox_proposals_list = model.simple_meta_test_pts(pts_feats, *data["img_metas"], rescale=False)

            bbox_proposals = [custom_bbox3d2result_output(LiDARInstance3DBoxes(bboxes.tensor[:, :7]), scores, labels, None) for bboxes, scores, labels in bbox_proposals_list]
            bbox_output = [custom_bbox3d2result_output(LiDARInstance3DBoxes(bboxes.tensor[:, :7]), scores, labels, None) for bboxes, scores, labels in bbox_output]

    
    return bbox_output, bbox_proposals, data, intensities


def aptiv_inference_detector(model, pcd):
        """Inference point cloud with the detector.
        Args:
            model (nn.Module): The loaded detector.
            pcd (str): Point cloud files.
        Returns:
            tuple: Predicted results and data from pipeline.
        """
        cfg = model.cfg
        device = next(model.parameters()).device  # model device

        if not isinstance(pcd, str):
            cfg = cfg.copy()
            # set loading pipeline type
            cfg.data.test.pipeline[0].type = 'LoadPointsFromDict'


        # build the data pipeline
        test_pipeline = deepcopy(cfg.data.test.pipeline)
        test_pipeline = Compose(test_pipeline)

        box_type_3d, box_mode_3d = get_box_type(cfg.data.test.box_type_3d)

        # load from point clouds file
        data = dict(
            pts_filename=pcd,
            box_type_3d=box_type_3d,
            box_mode_3d=box_mode_3d,
            # for ScanNet demo we need axis_align_matrix
            ann_info=dict(axis_align_matrix=np.eye(4)),
            sweeps=[],
            # set timestamp = 0
            timestamp=[0],
            img_fields=[],
            bbox3d_fields=[],
            pts_mask_fields=[],
            pts_seg_fields=[],
            bbox_fields=[],
            mask_fields=[],
            seg_fields=[]
        )

        data = test_pipeline(data)
        data = collate([data], samples_per_gpu=1)

        if next(model.parameters()).is_cuda:
            # scatter to specified GPU
            data = scatter(data, [device.index])[0]
        else:
            # this is a workaround to avoid the bug of MMDataParallel
            data['img_metas'] = data['img_metas'][0].data
            data['points'] = data['points'][0].data
            
        with torch.no_grad():
            if "mvx_faster_rcnn" in str(model.__class__).lower():
                img_feats, pts_feats = model.extract_feat(*data["points"], img=None, img_metas=data["img_metas"])
                bbox_list = [dict() for i in range(len(data["img_metas"]))]

                outs = model.pts_bbox_head(pts_feats)
                bbox_output, bbox_proposals_list = model.pts_bbox_head.get_meta_bboxes(*outs, *data["img_metas"], rescale=False)

                bbox_proposals = [custom_bbox3d2result_proposals(LiDARInstance3DBoxes(bboxes.tensor[:, :7]), scores[:, :5], dir_scores) for bboxes, scores, dir_scores in bbox_proposals_list]
                bbox_output = [custom_bbox3d2result_output(LiDARInstance3DBoxes(bboxes.tensor[:, :7]), scores, labels, dir_scores) for bboxes, scores, labels, dir_scores in bbox_output]

                scores_bool = np.max(bbox_proposals[0]["scores_3d"].numpy(), axis=1) > 0.1

                bbox_proposals = [
                    {
                        "boxes_3d": bbox_proposals[0]["boxes_3d"][scores_bool],
                        "scores_3d": bbox_proposals[0]["scores_3d"][scores_bool],
                        "dir_scores_3d": bbox_proposals[0]["dir_scores_3d"][scores_bool]
                    }
                ]
            elif "centerpoint" in str(model.__class__).lower():
                img_feats, pts_feats = model.extract_feat(*data["points"], img=None, img_metas=data["img_metas"])
                bbox_list = [dict() for i in range(len(data["img_metas"]))]

                bbox_output, bbox_proposals_list = model.simple_meta_test_pts(pts_feats, *data["img_metas"], rescale=False)

                bbox_proposals = [custom_bbox3d2result_output(LiDARInstance3DBoxes(bboxes.tensor[:, :7]), scores, labels, None) for bboxes, scores, labels in bbox_proposals_list]
                bbox_output = [custom_bbox3d2result_output(LiDARInstance3DBoxes(bboxes.tensor[:, :7]), scores, labels, None) for bboxes, scores, labels in bbox_output]
            
            
        return bbox_output, bbox_proposals, data
