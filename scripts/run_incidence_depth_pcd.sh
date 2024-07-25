CUDA_VISIBLE_DEVICES='1' python tools/run_incidence_depth_pcd.py \
--input_rgb_dir 'images' \
--output_dir 'output' \
--denoise_steps 10 \
--mode 'depth' \
--checkpoint 'checkpoint/stable-diffusion-2-1-marigold-8inchannels' \
--unet_ckpt_path "checkpoint/diffcalib-pcd"