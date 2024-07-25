import torch
import numpy as np

def align_scale_shift(pred: torch.tensor, target: torch.tensor):
    assert len(pred.shape) == 4
    assert pred.shape == target.shape

    bs, n, h, w = pred.shape

    for i in range(bs):
        mask = target[i] > 0
        target_mask = target[i][mask].detach().cpu().numpy()
        pred_mask = pred[i][mask].detach().cpu().numpy()
        if torch.sum(mask) > 10:
            scale, shift = np.polyfit(pred_mask, target_mask, deg=1)
            if scale < 0:
                scale = torch.median(target[i][mask]) / (torch.median(pred[i][mask]).detach() + 1e-8)
                shift = 0
        else:
            scale = 1
            shift = 0
        pred[i] = pred[i] * scale + shift

    return pred

def intrinsics_list2mat(intrinsics: torch.tensor) -> torch.tensor:
    intrinsics_mat = torch.zeros((intrinsics.shape[0], 3, 3)).float().to(intrinsics.device)
    intrinsics_mat[:, 0, 0] = intrinsics[:, 0]
    intrinsics_mat[:, 1, 1] = intrinsics[:, 1]
    intrinsics_mat[:, 0, 2] = intrinsics[:, 2]
    intrinsics_mat[:, 1, 2] = intrinsics[:, 3]
    intrinsics_mat[:, 2, 2] = 1.0
    return intrinsics_mat
