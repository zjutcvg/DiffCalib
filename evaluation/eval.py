# Script for inference on (in-the-wild) images

# Author: Bingxin Ke
# Last modified: 2023-12-18


import argparse
import os
import os.path as osp
import json
from glob import glob
import logging
import cv2

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
import time
import torch.nn.functional as F
from torchvision import transforms
from datasets import load_dataset
import torch.distributed as dist


from marigold import MarigoldPipeline
from marigold.util.seed_all import seed_all
from diffusers import UNet2DConditionModel, AutoencoderKL
from diffusers.schedulers import DDPMScheduler
from transformers import CLIPVisionModelWithProjection, CLIPTokenizer

# from accelerate import PartialState  # Can also be Accelerator or AcceleratorState
# from diffusers import DiffusionPipeline
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed

# Evaluation of depth
from util_depth.unproj_pcd import reconstruct_pcd, save_point_cloud
from avg_meter import MetricAverageMeter, NormalMeter
from util_depth.util import load_data, align_scale_shift, save_val_imgs, resize_depth_preserve
from util_depth.dataset import DepthJsonDataset

# Evaluation of surface normal
from util_normal.util import create_dataset_loader # , accumulate_prediction_error, compute_surface_normal_angle_error, log_normal_stats

from util_feature.dataset import RGBPathDataset

from marigold.models import DPTHead, CustomUNet2DConditionModel
import matplotlib.pyplot as plt
from marigold.marigold_pipeline import MarigoldDepthOutput
from marigold.util.image_util import ResizeLongestEdge

EXTENSION_LIST = [".jpg", ".jpeg", ".png"]

def setup_logger(logger_name, log_file, level=logging.INFO):
    log_setup = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fileHandler = logging.FileHandler(log_file, mode='a')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    log_setup.setLevel(level)
    log_setup.addHandler(fileHandler)
    log_setup.addHandler(streamHandler)

def check_list_path_exists(list_path):
    exist_flag = True
    for path in list_path:
        if not osp.exists(path):
            exist_flag = False
        else:
            pass
    
    return exist_flag

if "__main__" == __name__:
    logging.basicConfig(level=logging.INFO)

    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(
        description="Run single-image depth estimation using Marigold."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="Bingxin/Marigold",
        help="Checkpoint path or hub name.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=['KITTI', 'NYU', 'ScanNet', 'DIODE', 'ETH3D', 'Hypersim', 'in-the-wild'],
        help="Evaluation datasets name.",
    )
    parser.add_argument(
        "--gt_data_root",
        type=str,
        default='/mnt/nas/datasets2/MetricDepth',
        help="Path to gt_data.",
    )
    parser.add_argument(
        "--input_rgb_dir",
        type=str,
        default=None,
        help="Path to the input image folder if args.dataset == 'in-the-wild' .",
    )
    parser.add_argument(
        "--save_pcd",
        default=False,
        action="store_true",
        help="Save point cloud while evaluting depth.",
    )
    
    parser.add_argument(
        "--unet_ckpt_path",
        type=str,
        default=None,
        # default="./logs/logs_sd21_seg_pretrain/checkpoint-10000",
        help="Checkpoint path for unet.",
    )
    parser.add_argument(
        "--vae_ckpt_path",
        type=str,
        default=None,
        # default="./logs/logs_sd21_seg_pretrain/checkpoint-10000",
        help="Checkpoint path for unet.",
    )
    parser.add_argument(
        "--customized_head_ckpt_path",
        type=str,
        default=None,
        # default="./logs/logs_sd21_seg_pretrain/checkpoint-10000",
        help="Checkpoint path for customized head.",
    )
    parser.add_argument(
        "--customized_head_type",
        type=str,
        default=None,
        choices=['DPTHead'],
        # default="./logs/logs_sd21_seg_pretrain/checkpoint-10000",
        help="Type of customized head.",
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
        choices=['depth', 'seg', 'normal', 'feature'],
        default="depth",
        help="inference mode.",
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

    logging_dir = "logs"
    logging_dir = os.path.join(args.output_dir, logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision="fp16",
        log_with="tensorboard",
        project_config=accelerator_project_config,
    )

    checkpoint_path = args.checkpoint
    output_dir = osp.join(args.output_dir, args.dataset)
    os.makedirs(output_dir, exist_ok=True)
    setup_logger('eval', osp.join(output_dir, 'log.txt'))
    logger = logging.getLogger('eval')

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
    logger.info(f"output dir = {output_dir}")

    # -------------------- Device --------------------
    # if apple_silicon:
    #     if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    #         device = torch.device("mps:0")
    #     else:
    #         device = torch.device("cpu")
    #         logging.warning("MPS is not available. Running on CPU will be slow.")
    # else:
    #     if torch.cuda.is_available():
    #         device = torch.device("cuda")
    #     else:
    #         device = torch.device("cpu")
    #         logging.warning("CUDA is not available. Running on CPU will be slow.")

    device = accelerator.device
    logger.info(f"device = {device}")

    # -------------------- Data --------------------
    args.dataset = args.dataset
    # Depth Estimation
    if args.dataset == 'KITTI' and args.mode in ['depth']:
        dataset_root = osp.join(args.gt_data_root, 'kitti')
        gt_depth_scale = 256
        test_json = osp.join(dataset_root, 'test_annotation.json')
        dataset = DepthJsonDataset(test_json, dataset_root, processing_res=args.processing_res)
        # dataset = load_dataset("json", data_files=test_json, field="files")['train']
    elif args.dataset == 'NYU' and args.mode in ['depth']:
        dataset_root = osp.join(args.gt_data_root, 'nyu')
        gt_depth_scale = 1000
        test_json = osp.join(dataset_root, 'test_annotation.json')
        dataset = DepthJsonDataset(test_json, dataset_root, processing_res=args.processing_res)
        # dataset = load_dataset("json", data_files=test_json, field="files")['train']
    elif args.dataset == 'ScanNet' and args.mode in ['depth']:
        dataset_root = osp.join(args.gt_data_root, 'scannet_test')
        gt_depth_scale = 1000
        test_json = osp.join(dataset_root, 'test_annotation.json')
        dataset = DepthJsonDataset(test_json, dataset_root, processing_res=args.processing_res)
        # dataset = load_dataset("json", data_files=test_json, field="files")['train']
    elif args.dataset == 'DIODE' and args.mode in ['depth']:
        dataset_root = osp.join(args.gt_data_root, 'diode')
        gt_depth_scale = 1
        test_json = osp.join(dataset_root, 'test_annotation.json')
        dataset = DepthJsonDataset(test_json, dataset_root, processing_res=args.processing_res)
        # dataset = load_dataset("json", data_files=test_json, field="files")['train']
    elif args.dataset == 'ETH3D' and args.mode in ['depth']:
        dataset_root = osp.join(args.gt_data_root, 'ETH3D')
        gt_depth_scale = 1
        test_json = osp.join(dataset_root, 'test_annotations.json')
        dataset = DepthJsonDataset(test_json, dataset_root, processing_res=args.processing_res)
        # dataset = load_dataset("json", data_files=test_json, field="files")['train']
    elif args.dataset == 'Hypersim' and args.mode in ['depth']:
        dataset_root = '/test/xugk/code/marigold_github_repo/datasets_20231223'
        gt_depth_scale = 1
        test_json = osp.join(dataset_root, 'test_annotation.json')
        dataset = DepthJsonDataset(test_json, dataset_root, processing_res=args.processing_res)
    # Surface Normal Estimation
    elif args.dataset == 'ScanNet' and args.mode in ['normal']:
        dataset_root = osp.join(args.gt_data_root, 'scannet/scannet-frames')
        config_normal = {
            # 'ARCHITECTURE': 'dorn',
            # 'AUGMENTATION': '',
            'BATCH_SIZE': batch_size if batch_size != 0 else 1,
            # 'CKPT_PATH': '',
            # 'EVAL_ITER': 600,
            # 'LEARNING_RATE': 1e-3,
            # 'LOG_FOLDER': '',
            # 'MAX_EPOCH': 20,
            # 'PRINT_ITER': 60,
            # 'OPERATION': 'evaluate',
            # 'OPTIMIZER': 'adam',
            # 'RECTIFIED_CKPT_PATH': '',
            # 'SAVE_ITER': 6000,
            # 'SR_CKPT_PATH': './checkpoints/SR_only.ckpt',
            # 'TRAIN_DATASET': osp.join(args.gt_data_root, 'data/scannet_standard_train_test_val_split.pkl'),
            'TEST_DATASET': osp.join(args.gt_data_root, 'data/scannet_standard_train_test_val_split.pkl')}
    elif args.dataset == 'NYU' and args.mode in ['normal']:
        dataset_root = osp.join(args.gt_data_root, 'datasets/nyu-normal/')
        config_normal = {
            # 'ARCHITECTURE': 'dorn',
            # 'AUGMENTATION': '',
            'BATCH_SIZE': batch_size if batch_size != 0 else 1,
            # 'CKPT_PATH': '',
            # 'EVAL_ITER': 600,
            # 'LEARNING_RATE': 1e-3,
            # 'LOG_FOLDER': '',
            # 'MAX_EPOCH': 20,
            # 'PRINT_ITER': 60,
            # 'OPERATION': 'evaluate',
            # 'OPTIMIZER': 'adam',
            # 'RECTIFIED_CKPT_PATH': '',
            # 'SAVE_ITER': 6000,
            # 'SR_CKPT_PATH': './checkpoints/SR_only.ckpt',
            # 'TRAIN_DATASET': osp.join(args.gt_data_root, 'data/scannet_standard_train_test_val_split.pkl'),
            'TEST_DATASET': 'nyud'}
    elif args.dataset == 'in-the-wild':
        assert args.input_rgb_dir is not None
        rgb_paths = os.listdir(args.input_rgb_dir)
        rgb_paths = [osp.join(args.input_rgb_dir, path) for path in rgb_paths if (path.endswith('.jpg') or path.endswith('.png') or path.endswith('.JPG'))]

        if args.mode == 'feature':
            # NOTE: the image_transforms should match with the training process
            image_transforms = transforms.Compose( 
                [
                    transforms.Resize(args.processing_res, interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.CenterCrop(args.processing_res), # TODO: remove the crop and record the left coordinates for sdxl
                    # transforms.TrivialAugmentWide(),
                    # transforms.ColorJitter(),
                    transforms.ToTensor(), # -> [0, 1]
                    # transforms.Normalize([0.5], [0.5]), # -> [-1, 1]
                ]
            )
        else:
            image_transforms = transforms.Compose( 
                [
                    ResizeLongestEdge(processing_res, interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.ToTensor(), # -> [0, 1]
                ]
            )

        dataset = RGBPathDataset(rgb_paths, transform=image_transforms)
        dataset_root = './'
    else:
        raise ValueError

    # -------------------- Model --------------------
    if half_precision:
        dtype = torch.float16
        logger.info(f"Running with half precision ({dtype}).")
    else:
        dtype = torch.float32
    
    head_type = args.customized_head_type
    if head_type is not None:
        assert args.customized_head_ckpt_path is not None

        if head_type == 'DPTHead':
            customized_head = DPTHead.from_pretrained(args.customized_head_ckpt_path)
        else:
            raise NotImplementedError
    else:
        customized_head = None
    
    if args.unet_ckpt_path is not None:
        unet = CustomUNet2DConditionModel.from_pretrained(
            args.unet_ckpt_path, subfolder="unet", revision=args.non_ema_revision
        )
    else:
        unet = CustomUNet2DConditionModel.from_pretrained(
            checkpoint_path, subfolder="unet", revision=args.non_ema_revision
        )
    
    if args.vae_ckpt_path is not None:
        vae = AutoencoderKL.from_pretrained(
            args.vae_ckpt_path, subfolder="vae",
        )
    else:
        vae = AutoencoderKL.from_pretrained(
            checkpoint_path, subfolder="vae",
        )

    tokenizer = CLIPTokenizer.from_pretrained(
        checkpoint_path, subfolder="tokenizer",
    )

    marigold_params_ckpt = dict(
        torch_dtype=dtype,
        unet=unet,
        vae=vae,
        controlnet=None,
        text_embeds=None, # TODO: change it for sdxl model.
        image_projector=None, # TODO: change it for using image projector.
        customized_head=customized_head,
    )

    if args.image_encoder_ckpt_path is not None:
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            args.image_encoder_ckpt_path, subfolder="image_encoder", revision=args.non_ema_revision
        )
        marigold_params_ckpt['image_encoder'] = image_encoder
        marigold_params_ckpt['text_encoder'] = None
    else:
        marigold_params_ckpt['image_encoder'] = None
    
    # pipe = MarigoldPipeline.from_pretrained(checkpoint_path, **marigold_params_ckpt)
    pseudo_ckpt_path = 'checkpoint/stable-diffusion-2-1-marigold-8inchannels' # TODO: change it for your self. This is only pseudo ckpt, only for successful loading.
    assert osp.exists(pseudo_ckpt_path)
    pipe = MarigoldPipeline.from_pretrained(pseudo_ckpt_path, **marigold_params_ckpt)

    # pipe.scheduler = DDPMScheduler(
    #     beta_end=0.012,
    #     beta_schedule="scaled_linear",
    #     beta_start=0.00085,
    #     clip_sample=False,
    #     clip_sample_range=1.0,
    #     dynamic_thresholding_ratio=0.995,
    #     num_train_timesteps=1000,
    #     prediction_type="v_prediction",
    #     rescale_betas_zero_snr=False,
    #     sample_max_value=1.0,
    #     # set_alpha_to_one=False,
    #     steps_offset=1,
    #     thresholding=False,
    #     timestep_spacing="leading",
    #     trained_betas=None,
    # )

    # import pdb
    # pdb.set_trace()
    # DDIMScheduler {
    # "_class_name": "DDIMScheduler",
    # "_diffusers_version": "0.25.0.dev0",
    # "beta_end": 0.012,
    # "beta_schedule": "scaled_linear",
    # "beta_start": 0.00085,
    # "clip_sample": false,
    # "clip_sample_range": 1.0,
    # "dynamic_thresholding_ratio": 0.995,
    # "num_train_timesteps": 1000,
    # "prediction_type": "v_prediction",
    # "rescale_betas_zero_snr": false,
    # "sample_max_value": 1.0,
    # "set_alpha_to_one": false,
    # "skip_prk_steps": true,
    # "steps_offset": 1,
    # "thresholding": false,
    # "timestep_spacing": "leading",
    # "trained_betas": null
    # }

    # pipe.set_scheduler()

    # import pdb
    # pdb.set_trace()

    pipe = pipe.to(accelerator.device)
    pipe.set_progress_bar_config(disable=True)

    try:
        import xformers
        pipe.enable_xformers_memory_efficient_attention()
    except:
        pass  # run without xformers

    # pipe = pipe.to(device)
    
    # distributed_state = PartialState()
    # pipe.to(distributed_state.device)



    # ------------------------- Evaluator -------------------------
    if accelerator.num_processes == 1:
        is_distributed = False
    else:
        is_distributed = True
    if args.mode == 'depth':
        test_metrics=['abs_rel', 'rmse', 'delta1']
        dam_depth_global = MetricAverageMeter(test_metrics)
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size if batch_size != 0 else 1,
            shuffle=False,
            num_workers=batch_size if batch_size != 0 else 1,
        )
        dataloader = accelerator.prepare(dataloader)

        iterator = tqdm(
            dataloader,
            initial=0,
            desc="Steps",
            # Only show the progress bar once on each machine.
            disable=not accelerator.is_local_main_process,
        )

    elif args.mode == 'normal':
        test_metrics = ['mean', 'median', 'rmse', 'a1', 'a2', 'a3', 'a4', 'a5']
        dam_normal = NormalMeter(test_metrics)
        dataloader = create_dataset_loader(config_normal, dataset_root, processing_res=processing_res)
        dataloader = accelerator.prepare(dataloader)
        iterator = tqdm(
            dataloader,
            initial=0,
            desc="Steps",
            # Only show the progress bar once on each machine.
            disable=not accelerator.is_local_main_process,
        )
        # total_normal_errors = None

    elif args.mode == 'feature':
        feature_root = osp.join(output_dir, 'unet_features')
        os.makedirs(feature_root, exist_ok=True)
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size if batch_size != 0 else 1,
            shuffle=False,
            num_workers=batch_size if batch_size != 0 else 1,
        )
        dataloader = accelerator.prepare(dataloader)
        iterator = tqdm(
            dataloader,
            initial=0,
            desc="Steps",
            # Only show the progress bar once on each machine.
            disable=not accelerator.is_local_main_process,
        )
    else:
        raise ValueError

    processed_txt = osp.join(output_dir, 'processed_rgb_paths.txt')
    with open(processed_txt, 'w') as f:
        f.write('') # override the origin infos
    
    # cnt_xugk = torch.tensor(0).to(device)
    # -------------------- Inference and saving --------------------
    with torch.no_grad():
        os.makedirs(output_dir, exist_ok=True)

        for i, data in enumerate(iterator):
            torch.cuda.empty_cache()
            # time1 = time.time()
            rgb_paths = data['rgb']
            bs = len(rgb_paths)

            if not osp.isabs(rgb_paths[0]):
                rgb_paths = [osp.join(dataset_root, path) for path in rgb_paths]
            else:
                rgb_paths = rgb_paths
                
            # # Read input image
            # input_image = Image.open(rgb_path)
            # input_image_np = np.array(input_image)
                
            input_images_torch = data['images']

            # check file exists (only for depth now)
            depth_npy_save_paths = [os.path.join(output_dir_npy, os.path.splitext(rgb_path)[0].replace('/', '_') + "_pred_depth.npy") for rgb_path in rgb_paths]
            if args.mode == 'depth' and check_list_path_exists(depth_npy_save_paths):
                pred_depths = np.stack([np.load(path) for path in depth_npy_save_paths])
                pipe_out = MarigoldDepthOutput(depth_np=pred_depths, depth_colored=None, uncertainty=None)
            else:
                # Predict depth
                pipe_out = pipe(
                    input_images_torch,
                    denoising_steps=denoise_steps,
                    ensemble_size=ensemble_size,
                    processing_res=processing_res,
                    match_input_res=match_input_res,
                    batch_size=batch_size,
                    color_map=color_map,
                    show_progress_bar=True,
                    mode=args.mode,
                    rgb_paths=rgb_paths,
                    seed=seed,
                )

            # time3 = time.time()

            if args.mode == 'depth':

                vis_root = osp.join(output_dir, 'depth_vis')
                os.makedirs(vis_root, exist_ok=True)

                pred_depths = pipe_out.depth_np
                if len(pred_depths.shape) == 2:
                    pred_depths = pred_depths[None]
                for j in range(bs):
                    rgb_path = rgb_paths[j]
                    pred_depth = pred_depths[j]
                    input_image_torch = (input_images_torch[j] * 255)
                    input_image_np = input_image_torch.permute(1, 2, 0).detach().cpu().numpy()
                    input_image_np = input_image_np.astype(np.uint8)
                    
                    if args.dataset != 'in-the-wild':
                        gt_depth_path = data['depth'][j]
                        gt_depth_path = osp.join(dataset_root, gt_depth_path)
                        cam_in = data['cam_in'][j]

                    # Save as npy
                    rgb_name_base = os.path.splitext(rgb_path)[0].replace('/', '_')
                    pred_name_base = rgb_name_base + "_pred_depth"
                    npy_save_path = os.path.join(output_dir_npy, f"{pred_name_base}.npy")
                    if os.path.exists(npy_save_path):
                        logging.info(f"Numpy file '{npy_save_path}' exists, continue...")
                    else:
                        np.save(npy_save_path, pred_depth)

                    pred_depth = torch.from_numpy(pred_depth).to(device)
                    if args.dataset != 'in-the-wild':
                        gt_depth = load_data(gt_depth_path)
                        if args.dataset == 'KITTI':
                            # kb crop
                            height = gt_depth.shape[0]
                            width = gt_depth.shape[1]
                            top_margin = int(height - 352)
                            left_margin = int((width - 1216) / 2)
                            gt_depth = gt_depth[top_margin:(top_margin + 352), left_margin:(left_margin + 1216)]
                            cam_in[2] -= left_margin # cx
                            cam_in[3] -= top_margin # cy
                            # Eigen mask
                            gt_height, gt_width = gt_depth.shape
                            valid_mask = np.zeros_like(gt_depth).astype(bool)
                            valid_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height),
                                      int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = True
                            gt_depth[~valid_mask] = 0
                            # # grag mask
                            # valid_mask = torch.zeros_like(gt_depth).bool().to(device)
                            # valid_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height),
                            #           int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = True
                            # gt_depth[~valid_mask] = 0
                        if args.dataset == 'NYU':
                            # Eigen mask
                            gt_height, gt_width = gt_depth.shape
                            valid_mask = np.zeros_like(gt_depth).astype(bool)
                            valid_mask[45:471, 41:601] = True
                            gt_depth[~valid_mask] = 0
                        if args.dataset == 'DIODE':
                            gt_depth_mask = load_data(osp.join(dataset_root, data['depth_mask'][j]))
                            gt_depth[gt_depth_mask != 1] = 0
                        if args.dataset == 'ETH3D':
                            eth3d_resize_shape = (504, 756)
                            ratio_h = eth3d_resize_shape[0] / gt_depth.shape[0]
                            ratio_w = eth3d_resize_shape[1] / gt_depth.shape[1]
                            cam_in[0] *= ratio_w; cam_in[1] *= ratio_h; cam_in[2] *= ratio_w; cam_in[3] *= ratio_h
                            gt_depth = resize_depth_preserve(gt_depth, eth3d_resize_shape)
                            
                        gt_depth = gt_depth / gt_depth_scale
                        gt_depth = torch.from_numpy(gt_depth).to(device)
                        gt_depth[gt_depth > 300] = 0 # prevent extreme depth ranges.

                        pred_depth = F.interpolate(pred_depth[None, None], gt_depth.shape, mode='bilinear')[0, 0]
                        input_image_np = cv2.resize(input_image_np, (gt_depth.shape[1], gt_depth.shape[0]), interpolation=cv2.INTER_LINEAR)
                    
                        pred_global, _ = align_scale_shift(pred_depth, gt_depth)
                        mask = gt_depth > 0

                        with open(processed_txt, 'r') as f:
                            processed_rgbs = f.readlines()
                            processed_rgbs = [line.strip() for line in processed_rgbs]
                        if rgb_path not in processed_rgbs:
                            with open(processed_txt, 'a') as f:
                                f.write(rgb_path + '\n')
                            dam_depth_global.update_metrics_gpu(pred_global, gt_depth, mask, is_distributed)
                        else:
                            pass
                    else:
                        gt_depth = None

                    # time4 = time.time()
                    
                    # visualization
                    pred_norm = None; pred_norm_kappa = None; gt_norm = None
                    filename = rgb_path.replace('/', '_')
                    input_image_torch = torch.from_numpy(input_image_np)
                    # import pdb
                    # pdb.set_trace()
                    save_val_imgs(
                        i,
                        pred_depth,
                        gt_depth if gt_depth is not None else torch.ones_like(pred_depth, device=pred_depth.device),
                        pred_norm,
                        pred_norm_kappa,
                        gt_norm,
                        input_image_torch,
                        filename,
                        vis_root,
                    )

                    if args.save_pcd and args.dataset != 'in-the-wild':
                        pcd_root = osp.join(output_dir, 'depth_pcd')
                        os.makedirs(pcd_root, exist_ok=True)
                        gt_pcd_root = osp.join(output_dir, 'depth_pcd_gt')
                        os.makedirs(gt_pcd_root, exist_ok=True)

                        # pcd
                        pred_global = pred_global.detach().cpu().numpy().astype(np.float32)
                        pcd = reconstruct_pcd(pred_global.astype(np.float32), cam_in[0].item(), cam_in[1].item(), cam_in[2].item(), cam_in[3].item())
                        save_point_cloud(pcd.reshape((-1, 3)), input_image_np.reshape(-1, 3), osp.join(pcd_root, filename[:-4]+'.ply'))

                        # gt_pcd
                        gt_depth = gt_depth.detach().cpu().numpy()
                        gt_pcd = reconstruct_pcd(gt_depth, cam_in[0].item(), cam_in[1].item(), cam_in[2].item(), cam_in[3].item())
                        save_point_cloud(gt_pcd.reshape((-1, 3)), input_image_np.reshape(-1, 3), osp.join(gt_pcd_root, filename[:-4]+'.ply'))

                if args.dataset != 'in-the-wild':
                    eval_error_global = dam_depth_global.get_metrics()
                    logger.info('Depth Evaluation with Global Match :' + str(eval_error_global))

            elif args.mode == 'seg':
                seg_colored = pipe_out.seg_colored
                if len(seg_colored.shape) == 3:
                    seg_colored = seg_colored[None]
                for j in range(bs):
                    rgb_path = rgb_paths[j]
                    seg_colored_j = seg_colored[j]
                    input_image_torch = input_images_torch[j] * 255
                    input_image_np = input_image_torch.permute(1, 2, 0).detach().cpu().numpy()
                    input_image_np = input_image_np.astype(np.uint8)
                
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
                    # import pdb
                    # pdb.set_trace()
                    # semseg_path = os.path.join(semseg_dir, image_name).replace('_cam', '_scene_cam').replace('_frame.', '_geometry_preview_frame').replace('tonemap.', '')

                    seg_res = Image.fromarray(np.concatenate([input_image_np, np.array(seg_colored_j)],axis=1))
                    seg_res.save(colored_save_path)
                    # seg_colored.save(colored_save_path)
            elif args.mode == 'normal':
                vis_root = osp.join(output_dir, 'normal_vis')
                os.makedirs(vis_root, exist_ok=True)
                pred_norms = pipe_out.normal_np
                if len(pred_norms.shape) == 3:
                    pred_norms = pred_norms[None]

                for j in range(bs):
                    pred_norm = pred_norms[j]
                    input_image_torch = input_images_torch[j] * 255
                    input_image_np = input_image_torch.permute(1, 2, 0).detach().cpu().numpy()
                    input_image_np = input_image_np.astype(np.uint8)
                    rgb_path = rgb_paths[j]

                    pred_norm = torch.from_numpy(pred_norm).to(device)

                    if args.dataset != 'in-the-wild':
                        gt_norm = data['Z'][j].squeeze() # TODO: check the range of NYU dataset
                        gt_norm_mask = data['mask'][j].squeeze()

                        pred_norm = F.interpolate(pred_norm[None], (gt_norm.shape[-2], gt_norm.shape[-1]), mode='bilinear')[0]
                        input_image_np = cv2.resize(input_image_np, (gt_norm.shape[-1], gt_norm.shape[-2]), interpolation=cv2.INTER_LINEAR)
                    else:
                        gt_norm = None
                        gt_norm_mask = None
                    
                    pred_norm = pred_norm.flip(dims=[0]) # TODO: check the orientation of pred_norm, and think how to remove it?

                    if args.dataset != 'in-the-wild':
                        with open(processed_txt, 'r') as f:
                            processed_rgbs = f.readlines()
                            processed_rgbs = [line.strip() for line in processed_rgbs]
                        if rgb_path not in processed_rgbs:
                            with open(processed_txt, 'a') as f:
                                f.write(rgb_path + '\n')
                            dam_normal.update_metrics_gpu(pred_norm, gt_norm, (gt_norm_mask[None] > 0), is_distributed)
                            # cnt_xugk += 1
                        else:
                            pass
                    
                    # visualization
                    pred_depth = None; gt_depth = None; pred_norm_kappa = None
                    filename = rgb_path.replace('/', '_')
                    input_image_torch = torch.from_numpy(input_image_np)
                    save_val_imgs(
                        i,
                        pred_depth,
                        gt_depth,
                        pred_norm[None],
                        pred_norm_kappa,
                        gt_norm[None] if gt_norm is not None else torch.ones_like(pred_norm[None], device=pred_norm.device),
                        input_image_torch,
                        filename,
                        vis_root,
                    )

                if args.dataset != 'in-the-wild':
                    eval_error_normal = dam_normal.get_metrics()
                    logger.info('Surface Normal Evaliation :' + str(eval_error_normal))

            elif args.mode == 'feature':
                pred_features = pipe_out
                # import pdb
                # pdb.set_trace()
                for j in range(bs):
                    pred_feature = pred_features[j]
                    rgb_path = rgb_paths[j]

                    pred_feature = np.squeeze(pred_feature) # [steps, 4, H/8, W/8]
                    filename = osp.splitext(osp.basename(rgb_path))[0] + '.npy'
                    np.save(osp.join(feature_root, filename), pred_feature)
            else:
                raise NotImplementedError
        
        if args.mode == 'depth':
            eval_error_global = dam_depth_global.get_metrics()
            logger.info('Final Results of Depth Evaliation with Global Match :' + str(eval_error_global))
        elif args.mode == 'normal':
            eval_error_normal = dam_normal.get_metrics()
            # print('cnt_xugk_before :', cnt_xugk)
            # logger.info('cnt_xugk_before :' + str(cnt_xugk))
            # dist.all_reduce(cnt_xugk, op=dist.ReduceOp.SUM)
            # accelerator.wait_for_everyone() # TODO: check it? its effectiveness.
            # TODO: the cnt_xugk is not right here!!!
            # print('cnt_xugk_after :', cnt_xugk)
            # logger.info('cnt_xugk_after :' + str(cnt_xugk))
            # import pdb
            # pdb.set_trace()
            logger.info('Final Results of Surface Normal Evaliation :' + str(eval_error_normal))
        elif args.mode == 'feature':
            pass
        else:
            raise ValueError
        
    logger.info('Finished~')
