import os
import os.path as osp
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import matplotlib
from typing import Union
import h5py

img_file_type = ['.png', '.jpg', '.jpeg', '.bmp', '.tif']
np_file_type = ['.npz', '.npy']
JPG_file_type = ['.JPG']
hdf5_file_type = ['.hdf5']

def load_data(path: str, is_rgb_img: bool=False):
    if not osp.exists(path):
        raise RuntimeError(f'{path} does not exist.')

    data_type = osp.splitext(path)[-1]
    if data_type in img_file_type:
        if is_rgb_img:
            data = cv2.imread(path)
        else:
            data = cv2.imread(path, -1)
    elif data_type in np_file_type:
        data = np.load(path)
    elif data_type in JPG_file_type:
        # NOTE: only support .JPG file depth of ETH3D so far.
        if is_rgb_img:
            data = cv2.imread(path)
        else:
            f = open(path, 'r')
            data = np.fromfile(f, np.float32)
            data = data.reshape((4032, 6048))
    elif data_type in hdf5_file_type:
        data = filter_extreme_depth_values(convert_distance_to_depth(read_h5py(path)), percentage=2) # NOTE: only for hypersim
    else:
        raise RuntimeError(f'{data_type} is not supported in current version.')
    
    return data.squeeze()

def read_h5py(path):
    assert path.endswith('.hdf5')
    f = h5py.File(path, 'r')
    data = f['dataset'][:]
    return data

def convert_distance_to_depth(npyDistance):
    # convert distance to depth
    intWidth = 1024; intHeight = 768; fltFocal = 886.81
    npyImageplaneX = np.linspace((-0.5 * intWidth) + 0.5, (0.5 * intWidth) - 0.5, intWidth).reshape(1, intWidth).repeat(intHeight, 0).astype(np.float32)[:, :, None]
    npyImageplaneY = np.linspace((-0.5 * intHeight) + 0.5, (0.5 * intHeight) - 0.5, intHeight).reshape(intHeight, 1).repeat(intWidth, 1).astype(np.float32)[:, :, None]
    npyImageplaneZ = np.full([intHeight, intWidth, 1], fltFocal, np.float32)
    npyImageplane = np.concatenate([npyImageplaneX, npyImageplaneY, npyImageplaneZ], 2)
    npyDepth = npyDistance / np.linalg.norm(npyImageplane, 2, 2) * fltFocal
    return npyDepth

def filter_extreme_depth_values(depth, percentage=0):
    min_value = np.percentile(depth, percentage)
    max_value = np.percentile(depth, 100 - percentage)
    depth[depth > max_value] = max_value
    depth[depth < min_value] = min_value
    return depth

def resize_depth_preserve(depth, shape):
    """
    Resizes depth map preserving all valid depth pixels
    Multiple downsampled points can be assigned to the same pixel.

    Parameters
    ----------
    depth : np.array [h,w]
        Depth map
    shape : tuple (H,W)
        Output shape

    Returns
    -------
    depth : np.array [H,W,1]
        Resized depth map
    """
    # Store dimensions and reshapes to single column
    depth = np.squeeze(depth)
    h, w = depth.shape
    x = depth.reshape(-1)
    # Create coordinate grid
    uv = np.mgrid[:h, :w].transpose(1, 2, 0).reshape(-1, 2)
    # Filters valid points
    idx = x > 0
    crd, val = uv[idx], x[idx]
    # Downsamples coordinates
    crd[:, 0] = (crd[:, 0] * (shape[0] / h) + 0.5).astype(np.int32)
    crd[:, 1] = (crd[:, 1] * (shape[1] / w) + 0.5).astype(np.int32)
    # Filters points inside image
    idx = (crd[:, 0] < shape[0]) & (crd[:, 1] < shape[1])
    crd, val = crd[idx], val[idx]
    # Creates downsampled depth image and assigns points
    depth = np.zeros(shape)
    depth[crd[:, 0], crd[:, 1]] = val
    # Return resized depth map
    return depth

def align_scale_shift(pred: torch.tensor, target: torch.tensor):
    mask = target > 0
    target_mask = target[mask].cpu().numpy()
    pred_mask = pred[mask].cpu().numpy()
    if torch.sum(mask) > 10:
        scale, shift = np.polyfit(pred_mask, target_mask, deg=1)
        if scale < 0:
            scale = torch.median(target[mask]) / (torch.median(pred[mask]) + 1e-8)
            shift = 0
    else:
        scale = 1
        shift = 0
    pred = pred * scale + shift
    return pred, scale


def save_val_imgs(
    iter: int, 
    pred: Union[torch.tensor, None], 
    target: Union[torch.tensor, None], 
    normal: Union[torch.tensor, None],
    normal_kappa: Union[torch.tensor, None], 
    gt_normal: Union[torch.tensor, None], 
    rgb: torch.tensor, 
    filename: str, 
    save_dir: str,
    tb_logger=None
    ):
    """
    Save GT, predictions, RGB in the same file.
    """
    # import pdb
    # pdb.set_trace()
    rgb, pred_color, target_color, normal_color, normal_angular_error_color, gt_normal_color = get_data_for_log(pred, target, rgb, normal, normal_kappa, gt_normal)
    # rgb = rgb.transpose((1, 2, 0))
    # plt.imsave(os.path.join(save_dir, filename[:-4]+'_rgb.jpg'), rgb)
    # plt.imsave(os.path.join(save_dir, filename[:-4]+'_pred.png'), pred_scale, cmap='rainbow')
    # plt.imsave(os.path.join(save_dir, filename[:-4]+'_gt.png'), target_scale, cmap='rainbow')

    vis_elements = [rgb]

    if (pred_color is not None) and (target_color is not None):
        vis_elements.extend([pred_color, target_color])
    if (normal_color is not None) and (gt_normal_color is not None):
        vis_elements.extend([normal_color, gt_normal_color])
    if normal_angular_error_color is not None:
        vis_elements.extend([normal_angular_error_color])
    
    # import pdb
    # pdb.set_trace()
    cat_img = np.concatenate(vis_elements, axis=0)

    plt.imsave(os.path.join(save_dir, filename[:-4]+'_merge.jpg'), cat_img)

    # save to tensorboard
    if tb_logger is not None:
        tb_logger.add_image(f'{filename[:-4]}_merge.jpg', cat_img.transpose((2, 0, 1)), iter)

def get_data_for_log(pred: Union[torch.tensor, None], target: Union[torch.tensor, None], rgb: Union[torch.tensor, None], normal: Union[torch.tensor, None], normal_kappa: Union[torch.tensor, None], gt_normal: Union[torch.tensor, None]):

    rgb = rgb.squeeze().cpu().numpy()

    if (pred is not None) and (target is not None):
        pred = pred.squeeze().cpu().numpy()
        target = target.squeeze().cpu().numpy()
        
        pred[pred<0] = 0
        target[target<0] = 0
        pred_color = colorize_depth_maps_hwc(pred, pred.min(), pred.max(), cmap="Spectral")
        target_color = colorize_depth_maps_hwc(target, target.min(), target.max(), cmap="Spectral")
        pred_color = cv2.resize(pred_color, (rgb.shape[1], rgb.shape[0]))
        target_color = cv2.resize(target_color, (rgb.shape[1], rgb.shape[0]))
    else:
        pred_color = None
        target_color = None

    if (normal is not None) and (gt_normal is not None):
        normal = normal.cpu().permute(0, 2, 3, 1).numpy()
        gt_normal = gt_normal.cpu().permute(0, 2, 3, 1).numpy()
        normal_color = norm_to_rgb(normal)
        gt_normal_color = norm_to_rgb(gt_normal)
    else:
        normal_color = None
        gt_normal_color = None
    
    if normal_kappa is not None:
        normal_kappa = normal_kappa.cpu().permute(0, 2, 3, 1).numpy()
        normal_angular_error_color = kappa_to_alpha(normal_kappa)
        normal_angular_error_color[normal_angular_error_color > 60] = 60
        normal_angular_error_color[normal_angular_error_color < 0] = 0
        normal_angular_error_color = colorize_depth_maps_hwc(np.squeeze(normal_angular_error_color), normal_angular_error_color.min(), normal_angular_error_color.max(), cmap="Spectral")
    else:
        normal_angular_error_color = None

    return rgb, pred_color, target_color, normal_color, normal_angular_error_color, gt_normal_color

# normal vector to rgb values
def norm_to_rgb(norm):
    # norm: (B, H, W, 3)
    norm_rgb = ((norm[0, ...] + 1) * 0.5) * 255
    norm_rgb = np.clip(norm_rgb, a_min=0, a_max=255)
    norm_rgb = norm_rgb.astype(np.uint8)
    return norm_rgb

# kappa to exp error (only applicable to AngMF distribution)
def kappa_to_alpha(pred_kappa):
    alpha = ((2 * pred_kappa) / ((pred_kappa ** 2.0) + 1)) \
            + ((np.exp(- pred_kappa * np.pi) * np.pi) / (1 + np.exp(- pred_kappa * np.pi)))
    alpha = np.degrees(alpha)
    return alpha

def chw2hwc(chw):
    assert 3 == len(chw.shape)
    if isinstance(chw, torch.Tensor):
        hwc = torch.permute(chw, (1, 2, 0))
    elif isinstance(chw, np.ndarray):
        hwc = np.moveaxis(chw, 0, -1)
    return hwc

def colorize_depth_maps_hwc(
    depth_map, min_depth, max_depth, cmap="Spectral", valid_mask=None
):
    """
    Colorize depth maps.
    """
    assert len(depth_map.shape) >= 2, "Invalid dimension"

    if isinstance(depth_map, torch.Tensor):
        depth = depth_map.detach().clone().squeeze().numpy()
    elif isinstance(depth_map, np.ndarray):
        depth = np.squeeze(depth_map.copy())
    # reshape to [ (B,) H, W ]
    if depth.ndim < 3:
        depth = depth[np.newaxis, :, :]

    # colorize
    cm = matplotlib.colormaps[cmap]
    depth = ((depth - min_depth) / (max_depth - min_depth)).clip(0, 1)
    img_colored_np = cm(depth, bytes=False)[:, :, :, 0:3]  # value from 0 to 1
    img_colored_np = np.rollaxis(img_colored_np, 3, 1)

    if valid_mask is not None:
        if isinstance(depth_map, torch.Tensor):
            valid_mask = valid_mask.detach().numpy()
        valid_mask = np.squeeze(valid_mask)  # [H, W] or [B, H, W]
        if valid_mask.ndim < 3:
            valid_mask = valid_mask[np.newaxis, np.newaxis, :, :]
        else:
            valid_mask = valid_mask[:, np.newaxis, :, :]
        valid_mask = np.repeat(valid_mask, 3, axis=1)
        img_colored_np[~valid_mask] = 0

    if isinstance(depth_map, torch.Tensor):
        img_colored = torch.from_numpy(img_colored_np).float().squeeze()
    elif isinstance(depth_map, np.ndarray):
        img_colored = np.squeeze(img_colored_np)
    
    # import pdb
    # pdb.set_trace()
    img_colored = (img_colored * 255).astype(np.uint8)
    img_colored_hwc = chw2hwc(img_colored)

    return img_colored_hwc


