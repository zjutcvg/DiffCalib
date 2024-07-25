import argparse
import io
import os
import sys
pythonpath = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0,pythonpath)

from tools.calibrator import MonocularCalibrator
from tools.evaluation import compute_intrinsic_measure
from diffcalib.util.image_util import chw2hwc, colorize_depth_maps, resize_max_res, norm_to_rgb, resize_res
from glob import glob
import logging
from torchvision import transforms
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
import torch.nn as nn


from diffcalib.diffcalib_pipeline_rgb_12inchannels_depth import DiffCalibPipeline
from diffcalib.util.seed_all import seed_all
from diffcalib.test_utils import refine_focal, refine_shift

from diffcalib.spvcnn_classsification import SPVCNN_CLASSIFICATION
from diffusers import UNet2DConditionModel # AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, 
from diffusers.schedulers import DDPMScheduler, DDIMScheduler
from diffcalib.image_projector import ImageProjModel
from tools.tools import coords_gridN, resample_rgb, apply_augmentation

from transformers import CLIPVisionModelWithProjection

import time
import h5py
from plyfile import PlyData, PlyElement

EXTENSION_LIST = [".jpg", ".jpeg", ".png"]
import matplotlib.pyplot as plt

def plot_arrow_on_image(rgb_image, arrow_field, arrow_scale=0.02, sparse_factor=20, save_path=None):
    """
    绘制箭头在RGB图像上
    
    参数:
    - rgb_image: 三通道RGB图像,PIL Image对象
    - arrow_field: 入射场,包含箭头方向的三维向量场,numpy数组形式,数据类型为float32,形状为(3, height, width)
    - arrow_scale: 箭头长度的缩放因子,默认为0.02
    - sparse_factor: 绘制箭头的稀疏程度,默认为20,值越大绘制越稀疏
    - save_path: 图像保存路径,默认为None,不保存图像
    """
    # 将PIL图像转换为numpy数组
    rgb_image = np.array(rgb_image)
    
    # 获取图像尺寸
    height, width, _ = rgb_image.shape
    
    # 创建新图像,使原图像和箭头能够叠加
    overlay_image = np.copy(rgb_image)
    
    # 箭头长度的缩放因子
    arrow_scale *= min(height, width)
    
    # 绘制箭头
    for y in range(0, height, sparse_factor):  # 调整绘制的稀疏程度
        for x in range(0, width, sparse_factor):
            dx, dy, _ = arrow_field[:, y, x]  # 获取箭头方向
            # 绘制箭头
            plt.arrow(x, y, dx * arrow_scale, dy * arrow_scale, color='lightgreen', head_width=10, head_length=15)
    
    # 将箭头叠加到原始RGB图像上
    plt.imshow(overlay_image)
    plt.axis('off')  # 关闭坐标轴
    
    # 保存图像
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    else:
        plt.show()


def load_ckpt(args, shift_model=None, focal_model=None, scale_model=None):
    """
    Load checkpoint.
    """
    ckpt='./res101.pth'
    if os.path.isfile(ckpt):
        print("loading checkpoint res101.pth")
        checkpoint = torch.load(ckpt)
        if shift_model is not None:
            shift_model.load_state_dict(strip_prefix_if_present(checkpoint['shift_model'], 'module.'),
                                    strict=True)
        if focal_model is not None:
            focal_model.load_state_dict(strip_prefix_if_present(checkpoint['focal_model'], 'module.'),
                                    strict=True)
        if scale_model is not None:
            scale_model.load_state_dict(strip_prefix_if_present(scale_model_checkpoint['model_state_dict'], 'scale_model.'),
                                    strict=True)
        del checkpoint
        torch.cuda.empty_cache()
        # return shift_model

def get_pcd_base(H, W, u0, v0, fx, fy):
    x_row = np.arange(0, W)
    x = np.tile(x_row, (H, 1))
    x = x.astype(np.float32)
    u_m_u0 = x - u0

    y_col = np.arange(0, H)  # y_col = np.arange(0, height)
    y = np.tile(y_col, (W, 1)).T
    y = y.astype(np.float32)
    v_m_v0 = y - v0

    x = u_m_u0 / fx
    y = v_m_v0 / fy
    z = np.ones_like(x)
    pw = np.stack([x, y, z], axis=2)  # [h, w, c]
    return pw


def reconstruct_pcd(depth, fx, fy, u0, v0, pcd_base=None, mask=None):
    if type(depth) == torch.__name__:
        depth = depth.cpu().numpy().squeeze()
    # depth = cv2.medianBlur(depth, 5)
    if pcd_base is None:
        H, W = depth.shape
        pcd_base = get_pcd_base(H, W, u0, v0, fx, fy)
    pcd = depth[:, :, None] * pcd_base
    if mask:
        pcd[mask] = 0
    return pcd


def save_point_cloud(pcd, rgb, filename, binary=True):
    """Save an RGB point cloud as a PLY file.
    :paras
        @pcd: Nx3 matrix, the XYZ coordinates
        @rgb: Nx3 matrix, the rgb colors for each 3D point
    """
    assert pcd.shape[0] == rgb.shape[0]

    if rgb is None:
        gray_concat = np.tile(np.array([128], dtype=np.uint8),
                              (pcd.shape[0], 3))
        points_3d = np.hstack((pcd, gray_concat))
    else:
        points_3d = np.hstack((pcd, rgb))
    python_types = (float, float, float, int, int, int)
    npy_types = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'),
                 ('green', 'u1'), ('blue', 'u1')]
    if binary is True:
        # Format into Numpy structured array
        vertices = []
        for row_idx in range(points_3d.shape[0]):
            cur_point = points_3d[row_idx]
            vertices.append(
                tuple(
                    dtype(point)
                    for dtype, point in zip(python_types, cur_point)))
        vertices_array = np.array(vertices, dtype=npy_types)
        el = PlyElement.describe(vertices_array, 'vertex')

        # write
        PlyData([el]).write(filename)
    else:
        x = np.squeeze(points_3d[:, 0])
        y = np.squeeze(points_3d[:, 1])
        z = np.squeeze(points_3d[:, 2])
        r = np.squeeze(points_3d[:, 3])
        g = np.squeeze(points_3d[:, 4])
        b = np.squeeze(points_3d[:, 5])

        ply_head = 'ply\n' \
                    'format ascii 1.0\n' \
                    'element vertex %d\n' \
                    'property float x\n' \
                    'property float y\n' \
                    'property float z\n' \
                    'property uchar red\n' \
                    'property uchar green\n' \
                    'property uchar blue\n' \
                    'end_header' % r.shape[0]
        # ---- Save ply data to disk
        np.savetxt(filename, np.column_stack[x, y, z, r, g, b], fmt='%f %f %f %d %d %d', header=ply_head, comments='')


def reconstruct_3D(depth, f):
    """
    Reconstruct depth to 3D pointcloud with the provided focal length.
    Return:
        pcd: N X 3 array, point cloud
    """
    cu = depth.shape[1] / 2
    cv = depth.shape[0] / 2
    width = depth.shape[1]
    height = depth.shape[0]
    row = np.arange(0, width, 1)
    u = np.array([row for i in np.arange(height)])
    col = np.arange(0, height, 1)
    v = np.array([col for i in np.arange(width)])
    v = v.transpose(1, 0)

    if f > 1e5:
        print('Infinit focal length!!!')
        x = u - cu
        y = v - cv
        z = depth / depth.max() * x.max()
    else:
        x = (u - cu) * depth / f
        y = (v - cv) * depth / f
        z = depth

    x = np.reshape(x, (width * height, 1))
    y = np.reshape(y, (width * height, 1))
    z = np.reshape(z, (width * height, 1))
    pcd = np.concatenate((x, y, z), axis=1)
    pcd = pcd.astype(int)
    return pcd
def reconstruct_depth(depth, rgb, dir, pcd_name, focal):
    """
    para disp: disparity, [h, w]
    para rgb: rgb image, [h, w, 3], in rgb format
    """
    rgb = np.squeeze(rgb)
    depth = np.squeeze(depth)

    mask = depth < 1e-8
    depth[mask] = 0
    depth = depth / depth.max() * 10000

    pcd = reconstruct_3D(depth, f=focal)
    rgb_n = np.reshape(rgb, (-1, 3))
    save_point_cloud(pcd, rgb_n, os.path.join(dir, pcd_name + '.ply'))
def init_image_coor(height, width, u0=None, v0=None):
    u0 = width / 2.0 if u0 is None else u0
    v0 = height / 2.0 if v0 is None else v0

    x_row = np.arange(0, width)
    x = np.tile(x_row, (height, 1))
    x = x.astype(np.float32)
    u_u0 = x - u0

    y_col = np.arange(0, height)
    y = np.tile(y_col, (width, 1)).T
    y = y.astype(np.float32)
    v_v0 = y - v0
    return u_u0, v_v0

def make_shift_focallength_models():
    shift_model = SPVCNN_CLASSIFICATION(input_channel=3,
                                        num_classes=1,
                                        cr=1.0,
                                        pres=0.01,
                                        vres=0.01
                                        )
    focal_model = SPVCNN_CLASSIFICATION(input_channel=5,
                                        num_classes=1,
                                        cr=1.0,
                                        pres=0.01,
                                        vres=0.01
                                        )
    shift_model.eval()
    focal_model.eval()
    return shift_model, focal_model

def crop_image(image_path, output_paths):
    # 打开图片
    image = Image.open(image_path)
    
    # 获取图片的宽度和高度
    width, height = image.size

    # 计算裁剪后每个部分的大小
    crop_width = int(width * (2/3))
    crop_height = int(height * (2/3))

    # 裁剪并保存四个部分
    upper_left = image.crop((0, 0, crop_width, crop_height))
    upper_left.save(output_paths[0])

    upper_right = image.crop((width - crop_width, 0, width, crop_height))
    upper_right.save(output_paths[1])

    lower_right = image.crop((width - crop_width, height - crop_height, width, height))
    lower_right.save(output_paths[2])

    lower_left = image.crop((0, height - crop_height, crop_width, height))
    lower_left.save(output_paths[3])

if "__main__" == __name__:
    logging.basicConfig(level=logging.INFO)

    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(
        description="Run single-image depth estimation using diffcalib."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="ckpt",
        help="Checkpoint path or hub name.",
    )
    parser.add_argument(
        "--unet_ckpt_path",
        type=str,
        default=None,
        help="Checkpoint path for unet.",
    )
    parser.add_argument(
        "--scheduler_load_path",
        type=str,
        default=None,
        # default="./logs/logs_sd21_seg_pretrain/checkpoint-10000",
        help="Checkpoint path for scheduler.",
    )
    parser.add_argument(
        "--image_encoder_ckpt_path",
        type=str,
        default=None,
        # default="./logs/logs_sd21_seg_pretrain/checkpoint-10000",
        help="Checkpoint path for image_encoder.",
    )
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=['depth', 'seg', 'normal'],
        default="depth",
        help="inference mode.",
    )
    parser.add_argument(
        "--input_rgb_dir",
        type=str,
        required=True,
        help="Path to the input image folder.",
    )

    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory."
    )

    # inference setting
    parser.add_argument(
        "--denoise_steps",
        type=int,
        default=10,
        help="Diffusion denoising steps, more stepts results in higher accuracy but slower inference speed.",
    )
    parser.add_argument(
        "--ensemble_size",
        type=int,
        default=10,
        help="Number of predictions to be ensembled, more inference gives better results but runs slower.",
    )
    parser.add_argument(
        "--half_precision",
        action="store_true",
        help="Run with half-precision (16-bit float), might lead to suboptimal result.",
    )

    # resolution setting
    parser.add_argument(
        "--processing_res",
        type=int,
        default=768,
        help="Maximum resolution of processing. 0 for using input image resolution. Default: 768.",
    )
    parser.add_argument(
        "--output_processing_res",
        action="store_true",
        help="When input is resized, out put depth at resized operating resolution. Default: False.",
    )

    # depth map colormap
    parser.add_argument(
        "--color_map",
        type=str,
        default="Spectral",
        help="Colormap used to render depth predictions.",
    )

    # other settings
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=0,
        help="Inference batch size. Default: 0 (will be set automatically).",
    )
    parser.add_argument(
        "--apple_silicon",
        action="store_true",
        help="Flag of running on Apple Silicon.",
    )

    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    input_rgb_dir = args.input_rgb_dir
    output_dir = args.output_dir

    denoise_steps = args.denoise_steps
    ensemble_size = args.ensemble_size
    if ensemble_size > 15:
        logging.warning(f"Running with large ensemble size will be slow.")
    half_precision = args.half_precision

    processing_res = args.processing_res
    match_input_res = not args.output_processing_res

    color_map = args.color_map
    seed = args.seed
    batch_size = args.batch_size
    apple_silicon = args.apple_silicon
    if apple_silicon and 0 == batch_size:
        batch_size = 1  # set default batchsize

    # -------------------- Preparation --------------------
    # Random seed
    # import pdb
    # pdb.set_trace()
    if seed is None:
        import time
        seed = int(time.time())
        print('seed {}'.format(seed))

    seed_all(seed)

    # Output directories
    output_dir_color = os.path.join(output_dir, "{}_colored".format(args.mode))
    output_dir_tif = os.path.join(output_dir, "depth_bw")
    output_dir_npy = os.path.join(output_dir, "depth_npy")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir_color, exist_ok=True)
    os.makedirs(output_dir_tif, exist_ok=True)
    os.makedirs(output_dir_npy, exist_ok=True)
    logging.info(f"output dir = {output_dir}")

    # -------------------- Device --------------------
    if apple_silicon:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps:0")
        else:
            device = torch.device("cpu")
            logging.warning("MPS is not available. Running on CPU will be slow.")
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
            logging.warning("CUDA is not available. Running on CPU will be slow.")
    logging.info(f"device = {device}")

    # -------------------- Data --------------------

    datasetnames = ['Waymo']
    split = 'test'
    for datasetname in datasetnames:

        rgb_filename_list = glob(os.path.join('input/in-the-wild_example', "*"))
        rgb_filename_list = [
            f for f in rgb_filename_list if os.path.splitext(f)[1].lower() in EXTENSION_LIST
        ]
        rgb_filename_list = sorted(rgb_filename_list)
        n_images = len(rgb_filename_list)
        if n_images > 0:
            logging.info(f"Found {n_images} images")
        else:
            logging.error(f"No image found in '{input_rgb_dir}'")
            exit(1)

        # -------------------- Model --------------------
        if half_precision:
            dtype = torch.float16
            logging.info(f"Running with half precision ({dtype}).")
        else:
            dtype = torch.float32

        diffcalib_params_ckpt = dict(
            torch_dtype=dtype,
        )

        if args.unet_ckpt_path is not None:
            unet = UNet2DConditionModel.from_pretrained(
                args.unet_ckpt_path, subfolder="unet", revision=args.non_ema_revision
            )
            diffcalib_params_ckpt['unet'] = unet

        if args.scheduler_load_path is not None:
            diffcalib_params_ckpt['scheduler'] = DDIMScheduler.from_pretrained(args.scheduler_load_path, subfolder="scheduler")
        else:
            diffcalib_params_ckpt['scheduler'] = DDIMScheduler.from_pretrained(checkpoint_path, subfolder="scheduler")
        
        pipe = DiffCalibPipeline.from_pretrained(checkpoint_path, **diffcalib_params_ckpt)

        shift_model, focal_model = make_shift_focallength_models()
        # load checkpoint
        load_ckpt(args)
        shift_model.cuda()
        focal_model.cuda()
 
        try:
            import xformers
            pipe.enable_xformers_memory_efficient_attention()
        except:
            pass  # run without xformers

        # if pipe.image_projector is not None:
        #     pipe.image_projector.to(device)
        pipe = pipe.to(device)

        # -------------------- Inference and saving --------------------
        with torch.no_grad():
            # prepare dic for estimatation
            os.makedirs(output_dir, exist_ok=True)
            measurements = torch.zeros(n_images, 6).to(device)
            world_size = 1 # now just one img
            totensor = transforms.ToTensor()
            normalize = transforms.Normalize(mean=0.5, std=0.5)

            image_dir_out = 'output'
            os.makedirs(image_dir_out, exist_ok=True)
            for id, rgb_path in enumerate(tqdm(rgb_filename_list, desc="Estimating {}".format(args.mode), leave=True)):
                stem_name = rgb_path.split('/')[-1]
            
                
                rgb = Image.open(rgb_path)
               
                
                rgb = rgb.resize((rgb.width, rgb.height))
                # rgb = rgb.resize((rgb.width//3, rgb.height//3))
                # rgb1 = rgb1.resize((768,768))
                rgb = rgb.convert("RGB")
                
                raw_rgb = np.array(rgb)
                w, h = rgb.size

                # Step 1 : Resize
                rgb = rgb.resize((processing_res, processing_res))

                raw = rgb
               
                # Normalization
                rgb = normalize(totensor(rgb))
                # rgb = normalize(rgb)

                T = torch.eye(3, dtype=torch.float32)

                # Predict depth
                pipe_out = pipe(
                    rgb,
                    size=(w,h),
                    denoising_steps=denoise_steps,
                    ensemble_size=ensemble_size,
                    processing_res=processing_res,
                    match_input_res=match_input_res,
                    batch_size=batch_size,
                    color_map=color_map,
                    show_progress_bar=True,
                    mode='depth',
                )
                # end_time = time.time()
                # generate_time = end_time - start_time
                image = pipe_out['incident_colored']
                incidence = pipe_out['incident_np']
                fig, ax = plt.subplots()
                ax.imshow(raw)

                # 设置箭头的密度,此处使箭头稀疏3倍
                step = 60

                # 提取X, Y分量
                est_U = incidence[0, ...]  # Estimated X分量
                est_V = incidence[1, ...]  # Estimated Y分量
                
                # 在图像上绘制箭头
                for i in range(0, est_U.shape[1], step):
                    for j in range(0, est_V.shape[0], step):
                        # 绘制箭头,可以自定义颜色和宽度等属性
                        ax.arrow(i, j, est_U[j, i]*76.8, est_V[j, i]*76.8, color='orange', head_width=3, head_length=6, length_includes_head=True)
        
                ax.axis('off')
                plt.savefig(f'{image_dir_out}/{stem_name}-v.jpg', bbox_inches='tight', pad_inches=0, dpi=300)

                depth_np = pipe_out['depth_np']
                depth_colored = pipe_out['depth_colored']
                depth_colored.save('{}/{}.jpg'.format(image_dir_out, stem_name))
                # np.save('example3.npy',depth_np)
                os.makedirs(image_dir_out, exist_ok=True)
                # recon_start_time = time.time()
                monocalibrator = MonocularCalibrator(l1_th=0.02)
                Kest = monocalibrator.calibrate_camera_4DoF(torch.tensor(incidence).unsqueeze(0).to(device), device, RANSAC_trial=2048)
                Kest = Kest.cpu().numpy()
                proposed_scaled_focal = Kest[0][0] * (w / 768.0)
                pred_depth_norm = depth_np - depth_np.min() + 0.5
                dmax = np.percentile(pred_depth_norm, 98)
                pred_depth_norm = pred_depth_norm / dmax
                cam_u0, cam_v0 = raw_rgb.shape[1] / 2.0, raw_rgb.shape[0] / 2.0

                shift_1 = refine_shift(pred_depth_norm, shift_model, proposed_scaled_focal, raw_rgb.shape[1] / 2.0,  raw_rgb.shape[0] / 2.0)
                depth_np = pred_depth_norm - shift_1.item()

                reconstruct_depth(depth_np, raw_rgb, image_dir_out, '{}-pcd'.format(stem_name), focal=proposed_scaled_focal)
