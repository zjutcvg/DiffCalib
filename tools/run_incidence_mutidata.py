import argparse
import io
import os
import sys
pythonpath = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0,pythonpath)
import pdb
# pdb.set_trace()
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

from diffcalib.diffcalib_pipeline_rgb_12inchannels import DiffcalibPipeline
from diffcalib.util.seed_all import seed_all
from diffusers import UNet2DConditionModel # AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, 
from diffusers.schedulers import DDPMScheduler, DDIMScheduler
from diffcalib.image_projector import ImageProjModel
from tools.tools import coords_gridN, resample_rgb, apply_augmentation

from transformers import CLIPVisionModelWithProjection

import time
import h5py

EXTENSION_LIST = [".jpg", ".jpeg", ".png"]


if "__main__" == __name__:
    logging.basicConfig(level=logging.INFO)

    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(
        description="Run single-image depth estimation using diffcalib."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoint/stable-diffusion-2-1-marigold-8inchannels",
        help="Checkpoint path or hub name.",
    )
    parser.add_argument(
        "--unet_ckpt_path",
        type=str,
        # default=None,
        default="checkpoint/best-22500-0.07517",
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
        choices=['depth', 'incident', 'normal'],
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
    name_mapping = {
            'ScanNet': 'scannet',
            'MegaDepth': 'megadepth',
            'NYUv2': 'nyuv2',
            'Cityscapes': 'cityscapes',
            'MVS': 'mvs',
            'RGBD': 'rgbd',
            'Scenes11': 'scenes11',
            'SUN3D': 'sun3d',
            'BIWIRGBDID': 'biwirgbdid',
            'CAD120': 'cad120',
            'KITTI': 'kitti',
            'Waymo': 'waymo',
            'Nuscenes': 'nuscenes',
            'ARKitScenes': 'arkitscenes',
            'Objectron': 'objectron',
            'MVImgNet': 'mvimgnet'
        }
    datasets_train = [
            'Nuscenes',
            'KITTI',
            'Cityscapes',
            'NYUv2',
            'ARKitScenes',
            'MegaDepth',
            'SUN3D',
            'MVImgNet',
            'Objectron',
        ]
    
    # datasetnames = ['RGBD', 'Waymo','Scenes11', 'ScanNet' ,'MVS','Nuscenes', 'KITTI','Cityscapes', 'NYUv2' , 'ARKitScenes','SUN3D','MVImgNet','Objectron']
    datasetnames = ['Waymo','RGBD', 'Scenes11', 'ScanNet' ,'MVS']
    split = 'test'

    for datasetname in datasetnames:
        split_path = os.path.join('splits', '{}_{}.txt'.format(name_mapping[datasetname], split))
        print(split_path)
        with open(split_path) as file:
            data_names = [line.rstrip() for line in file]
        val_len = len(data_names)

        if val_len > 0:
            logging.info(f"Found {val_len} images")
        else:
            logging.error(f"No image found in '{split_path}'")
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
        
        pipe = DiffcalibPipeline.from_pretrained(checkpoint_path, **diffcalib_params_ckpt)

        
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
            measurements = torch.zeros(val_len, 6).to(device)
            world_size = 1 # now just one img
            totensor = transforms.ToTensor()
            normalize = transforms.Normalize(mean=0.5, std=0.5)
            # for id, rgb_path in enumerate(tqdm(rgb_filename_list, desc="Estimating {}".format(args.mode), leave=True)):
            for id, data_name in enumerate(tqdm(data_names, desc="Estimating {}".format(args.mode), leave=True)):
                # load hdf5 24.2.14
                scene_name, stem_name = data_name.split(' ')
                h5pypath = os.path.join('/mnt/nas/share/home/xugk/hxk/code/marigold/data/MonoCalib', datasetname, '{}.hdf5'.format(scene_name))

                if not os.path.exists(h5pypath):
                    print("H5 file %s missing" % h5pypath)
                    assert os.path.exists(h5pypath)

                with h5py.File(h5pypath, 'r') as hf:
                    # Load Intrinsic
                    K_color = np.array(hf['intrinsic'][stem_name])
                    # Load RGB
                    rgb = Image.open(io.BytesIO(np.array(hf['color'][stem_name])))

                w, h = rgb.size
                # pdb.set_trace()

                # Step 1 : Resize
                rgb = rgb.resize((processing_res, processing_res))

                scaleM = np.eye(3)
                scaleM[0, 0] = processing_res / w
                scaleM[1, 1] = processing_res / h
                aspect_ratio_restoration = (scaleM[1, 1] / scaleM[0, 0]).item()

                K = torch.from_numpy(scaleM @ K_color).float()
                # Color Augmentation only in training
                # rgb = self.color_augmentation_fun(rgb) if self.coloraugmentation else rgb

                # Normalization
                rgb = normalize(totensor(rgb))
                
                T = torch.eye(3, dtype=torch.float32)

                pipe_out = pipe(
                    # validation_prompt, 
                    rgb,
                    denoising_steps=10,
                    mode='incident'
                )

                image = pipe_out['incident_colored']
                incidence = pipe_out['incident_np']

                w, h = incidence.shape[0], incidence.shape[1]

                if args.mode == 'depth':

                    depth_pred: np.ndarray = pipe_out.depth_np
                    depth_colored: Image.Image = pipe_out.depth_colored

                    # Save as npy
                    rgb_name_base = os.path.splitext(os.path.basename(rgb_path))[0]
                    pred_name_base = rgb_name_base + "_depth_pred"
                    npy_save_path = os.path.join(output_dir_npy, f"{pred_name_base}.npy")
                    if os.path.exists(npy_save_path):
                        logging.warning(f"Existing file: '{npy_save_path}' will be overwritten")
                    np.save(npy_save_path, depth_pred)

                    # Save as 16-bit uint png
                    depth_to_save = (depth_pred * 65535.0).astype(np.uint16)
                    png_save_path = os.path.join(output_dir_tif, f"{pred_name_base}.png")
                    if os.path.exists(png_save_path):
                        logging.warning(f"Existing file: '{png_save_path}' will be overwritten")
                    Image.fromarray(depth_to_save).save(png_save_path, mode="I;16")

                    # Colorize
                    colored_save_path = os.path.join(
                        output_dir_color, f"{pred_name_base}_colored.png"
                    )
                    if os.path.exists(colored_save_path):
                        logging.warning(
                            f"Existing file: '{colored_save_path}' will be overwritten"
                        )
                    depth_colored.save(colored_save_path)

                elif args.mode == 'seg':
                    seg_colored: Image.Image = pipe_out.seg_colored
                    rgb_name_base = os.path.splitext(os.path.basename(rgb_path))[0]
                    pred_name_base = rgb_name_base + "_seg_pred"
                    # Colorize
                    colored_save_path = os.path.join(
                        output_dir_color, f"{pred_name_base}_steps_{denoise_steps}_colored.png"
                    )
                    if os.path.exists(colored_save_path):
                        logging.warning(
                            f"Existing file: '{colored_save_path}' will be overwritten"
                        )

                    seg_res = Image.fromarray(np.concatenate([np.array(input_image), np.array(seg_colored)],axis=1))
                    seg_res.save(colored_save_path)
                    # seg_colored.save(colored_save_path)
                elif args.mode == 'incident':
                    # color img
                    seg_colored: Image.Image = image
                    rgb_name_base = os.path.splitext(os.path.basename(stem_name))[0]
                    pred_name_base = rgb_name_base + "_incident_pred_with_image"
                    # Colorize
                    colored_save_path = os.path.join(
                        output_dir_color, f"{pred_name_base}_colored.png"
                    )
                    if os.path.exists(colored_save_path):
                        logging.warning(
                            f"Existing file: '{colored_save_path}' will be overwritten"
                        )
                    seg_res = Image.fromarray(np.array(seg_colored))
                    
                    monocalibrator = MonocularCalibrator(l1_th=0.02)
                    Kgt, Kest = torch.tensor(K).unsqueeze(0), monocalibrator.calibrate_camera_4DoF(torch.tensor(incidence).unsqueeze(0).to(device), device, RANSAC_trial=2048)
                    
                    print(f'Kgt:{Kgt}')
                    print(f'Kest:{Kest}')
                    
                    error_fx, error_fy, error_f, error_bx, error_by, error_b = compute_intrinsic_measure(
                        Kest=torch.clone(Kest),
                        Kgt=torch.clone(Kgt.squeeze()),
                        h=768,
                        w=768
                    )
                    print(f'error_fx: {error_fx:.4f}, error_fy: {error_fy:.4f}, error_f: {error_f:.4f}')
                    print(f'error_bx: {error_bx:.4f}, error_by: {error_by:.4f}, error_b: {error_b:.4f}')
                    
                    error_all = np.array([error_fx, error_fy, error_f, error_bx, error_by, error_b])
                    error_all = torch.from_numpy(error_all).float().to(device)

                    measurements[id * world_size + 0] += error_all
                else:
                    raise NotImplementedError
                
            if args.mode == 'incident':
                zero_entry = torch.sum(measurements.abs(), dim=1) == 0
                assert torch.sum(zero_entry) == 0

                measurements = torch.mean(measurements, dim=0)
                measurements = {
                    'error_fx': measurements[0].item(),
                    'error_fy': measurements[1].item(),
                    'error_f': measurements[2].item(),
                    'error_bx': measurements[3].item(),
                    'error_by': measurements[4].item(),
                    'error_b': measurements[5].item()
                }
                print(measurements)
                # 打开文件以追加模式
                with open('incidence_eval_best.txt', 'a') as f:
                    f.write(f'{datasetname}: \n')
                    # 遍历字典中的键值对，将其写入文件中
                    for key, value in measurements.items():
                        f.write(f'{key}: {value}\n')