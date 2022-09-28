import torch

def custom_bbox3d2result_proposals(bboxes, scores, dir_scores, attrs=None):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (torch.Tensor):     Bounding boxes with shape (N, 5).
        labels (torch.Tensor):     Labels with shape (N, ).
        scores (torch.Tensor):     Scores with shape (N, ).
        dir_scores (torch.Tensor): Direction Scores with shape (N, ).
        attrs (torch.Tensor, optional): Attributes with shape (N, ).
            Defaults to None.

    Returns:
        dict[str, torch.Tensor]: Bounding box results in cpu mode.

            - boxes_3d (torch.Tensor): 3D boxes.
            - scores (torch.Tensor): Prediction scores.
            - labels_3d (torch.Tensor): Box labels.
            - dir_scores (torch.Tensor): Direction Scores.
            - attrs_3d (torch.Tensor, optional): Box attributes.
    """
    result_dict = dict(
        boxes_3d=bboxes.to('cpu') if bboxes is not None else None,
        scores_3d=scores.cpu() if scores is not None else None,
        dir_scores_3d=dir_scores.cpu() if dir_scores is not None else torch.tensor([0] * len(scores))
    )

    if attrs is not None:
        result_dict['attrs_3d'] = attrs.cpu()

    return result_dict


def custom_bbox3d2result_output(bboxes, scores, labels, dir_scores, attrs=None):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (torch.Tensor):     Bounding boxes with shape (N, 5).
        labels (torch.Tensor):     Labels with shape (N, ).
        scores (torch.Tensor):     Scores with shape (N, ).
        dir_scores (torch.Tensor): Direction Scores with shape (N, ).
        attrs (torch.Tensor, optional): Attributes with shape (N, ).
            Defaults to None.

    Returns:
        dict[str, torch.Tensor]: Bounding box results in cpu mode.

            - boxes_3d (torch.Tensor): 3D boxes.
            - scores (torch.Tensor): Prediction scores.
            - labels_3d (torch.Tensor): Box labels.
            - dir_scores (torch.Tensor): Direction Scores.
            - attrs_3d (torch.Tensor, optional): Box attributes.
    """
    result_dict = dict(
        boxes_3d=bboxes.to('cpu') if bboxes is not None else None,
        scores_3d=scores.cpu() if scores is not None else None,
        labels_3d=labels.cpu() if labels is not None else None,
        dir_scores_3d=dir_scores.cpu() if dir_scores is not None else torch.tensor([0] * len(scores))
    )

    if attrs is not None:
        result_dict['attrs_3d'] = attrs.cpu()

    return result_dict