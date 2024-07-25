from typing import List, Dict, Union
import pdb
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm.auto import tqdm
from PIL import Image
from torchvision import transforms

from diffusers import (
    DiffusionPipeline,
    DDIMScheduler,
    UNet2DConditionModel,
    AutoencoderKL,
)
from diffusers.utils import BaseOutput
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor

from .util.image_util import chw2hwc, colorize_depth_maps, resize_max_res, norm_to_rgb, resize_res
from .util.batchsize import find_batch_size
from .util.ensemble import ensemble_depths

class DiffCalibNormalOutput(BaseOutput):

    incident_np: np.ndarray
    incident_colored: Image.Image
    normal_np: np.ndarray
    normal_colored: Image.Image
    uncertainty: Union[None, np.ndarray]

class DiffCalibDepthOutput(BaseOutput):
    """
    Output class for DiffCalib monocular depth prediction pipeline.

    Args:
        depth_np (np.ndarray):
            Predicted depth map, with depth values in the range of [0, 1]
        depth_colored (PIL.Image.Image):
            Colorized depth map, with the shape of [3, H, W] and values in [0, 1]
        uncertainty (None` or `np.ndarray):
            Uncalibrated uncertainty(MAD, median absolute deviation) coming from ensembling.
    """
    incident_np: np.ndarray
    incident_colored: Image.Image
    depth_np: np.ndarray
    depth_colored: Image.Image
    uncertainty: Union[None, np.ndarray]


class DiffCalibSegOutput(BaseOutput):
    """
    Output class for DiffCalib monocular depth prediction pipeline.

    Args:
        depth_np (np.ndarray):
            Predicted depth map, with depth values in the range of [0, 1]
        depth_colored (PIL.Image.Image):
            Colorized depth map, with the shape of [3, H, W] and values in [0, 1]
        uncertainty (None` or `np.ndarray):
            Uncalibrated uncertainty(MAD, median absolute deviation) coming from ensembling.
    """
    # seg_np: np.ndarray
    seg_colored: Image.Image
    uncertainty: Union[None, np.ndarray]



class DiffCalibPipeline(DiffusionPipeline):
    """
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        unet (UNet2DConditionModel):
            Conditional U-Net to denoise the depth latent, conditioned on image latent.
        vae (AutoencoderKL):
            Variational Auto-Encoder (VAE) Model to encode and decode images and depth maps
            to and from latent representations.
        scheduler (DDIMScheduler):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents.
        text_encoder (CLIPTextModel):
            Text-encoder, for empty text embedding.
        tokenizer (CLIPTokenizer):
            CLIP tokenizer.
    """

    rgb_latent_scale_factor = 0.18215
    depth_latent_scale_factor = 0.18215
    seg_latent_scale_factor = 0.18215

    def __init__(
        self,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        scheduler: DDIMScheduler,
        # text_encoder: CLIPTextModel,
        # tokenizer: CLIPTokenizer,
    ):
        super().__init__()

        self.register_modules(
            unet=unet,
            vae=vae,
            scheduler=scheduler,
            # text_encoder=text_encoder,
            # tokenizer=tokenizer,
        )

        self.empty_text_embed = None
        self.__encode_empty_text()

    @torch.no_grad()
    def __call__(
        self,
        input_image: Image,
        size: (768, 768),
        denoising_steps: int = 10,
        # num_inference_steps: int = 10,
        ensemble_size: int = 10,
        processing_res: int = 768,
        match_input_res: bool = True,
        batch_size: int = 0,
        color_map: str = "Spectral",
        show_progress_bar: bool = True,
        ensemble_kwargs: Dict = None,
        mode: str = 'depth',
    ) -> DiffCalibDepthOutput:
        """
        Function invoked when calling the pipeline.

        Args:
            input_image (Image):
                Input RGB (or gray-scale) image.
            processing_res (int, optional):
                Maximum resolution of processing.
                If set to 0: will not resize at all.
                Defaults to 768.
            match_input_res (bool, optional):
                Resize depth prediction to match input resolution.
                Only valid if `limit_input_res` is not None.
                Defaults to True.
            denoising_steps (int, optional):
                Number of diffusion denoising steps (DDIM) during inference.
                Defaults to 10.
            ensemble_size (int, optional):
                Number of predictions to be ensembled.
                Defaults to 10.
            batch_size (int, optional):
                Inference batch size, no bigger than `num_ensemble`.
                If set to 0, the script will automatically decide the proper batch size.
                Defaults to 0.
            show_progress_bar (bool, optional):
                Display a progress bar of diffusion denoising.
                Defaults to True.
            color_map (str, optional):
                Colormap used to colorize the depth map.
                Defaults to "Spectral".
            ensemble_kwargs ()
        Returns:
            `DiffCalibDepthOutput`
        """

        device = self.device
        input_size = size

        if not match_input_res:
            assert (
                processing_res is not None
            ), "Value error: `resize_output_back` is only valid with "
        assert processing_res >= 0
        assert denoising_steps >= 1
        assert ensemble_size >= 1

        rgb_norm = input_image.to(device)
        assert rgb_norm.min() >= -1.0 and rgb_norm.max() <= 1.0

        # ----------------- Predicting depth -----------------
        # Batch repeated input image
        duplicated_rgb = torch.stack([rgb_norm] * ensemble_size)
        single_rgb_dataset = TensorDataset(duplicated_rgb)
        if batch_size > 0:
            _bs = batch_size
        else:
            _bs = find_batch_size(
                ensemble_size=ensemble_size, 
                input_res=max(rgb_norm.shape[1:]),
                dtype=self.dtype,
            )

        single_rgb_loader = DataLoader(
            single_rgb_dataset, batch_size=_bs, shuffle=False
        )

        # Predict depth maps (batched)
        incident_pred_ls = []
        normal_pred_ls = []
        # pdb.set_trace()
        if show_progress_bar:
            iterable = tqdm(
                single_rgb_loader, desc=" " * 2 + "Inference batches", leave=False
            )
        else:
            iterable = single_rgb_loader
        for batch in iterable:
            (batched_img,) = batch
            incident_pred_raw, normal_pred_raw = self.single_infer(
                rgb_in=batched_img,
                num_inference_steps=denoising_steps,
                show_pbar=show_progress_bar,
                mode=mode,
            )
            incident_pred_ls.append(incident_pred_raw.detach().clone())
            normal_pred_ls.append(normal_pred_raw.detach().clone())
        # pdb.set_trace()
        incident_preds = torch.concat(incident_pred_ls, axis=0).squeeze()
        depth_preds = torch.concat(normal_pred_ls, axis=0).squeeze()
        torch.cuda.empty_cache()  # clear vram cache for ensembling


        # ----------------- Test-time ensembling -----------------
        if mode=='depth':
            if ensemble_size > 1:
                depth_pred, pred_uncert = ensemble_depths(
                    depth_preds, **(ensemble_kwargs or {})
                )
            else:
                depth_pred = depth_preds
                pred_uncert = None
            # if match_input_res:
            #     import pdb
            #     pdb.set_trace()
            #     depth_pred = F.interpolate(depth_pred, input_size, mode='bilinear')

        # ----------------- Post processing -----------------
        if mode == 'depth':
            # Scale prediction to [0, 1]
            min_d = torch.min(depth_pred)
            max_d = torch.max(depth_pred)
            depth_pred = (depth_pred - min_d) / (max_d - min_d)

            # Convert to numpy
            depth_pred = depth_pred.cpu().numpy().astype(np.float32)

            # Resize back to original resolution
            if match_input_res:
                pred_img = Image.fromarray(depth_pred)
                pred_img = pred_img.resize(input_size)
                depth_pred = np.asarray(pred_img)
            # # import pdb
            # pdb.set_trace()
            # Clip output range
            depth_pred = depth_pred.clip(0, 1)
            # import pdb
            # pdb.set_trace()

            # Colorize
            depth_colored = colorize_depth_maps(
                depth_pred, 0, 1, cmap=color_map
            ).squeeze()  # [3, H, W], value in (0, 1)
            depth_colored = (depth_colored * 255).astype(np.uint8)
            depth_colored_hwc = chw2hwc(depth_colored)
            depth_colored_img = Image.fromarray(depth_colored_hwc)

            # incident
            incident_preds = incident_preds.mean(dim=0)
            incident = incident_preds.clip(-1, 1).cpu().numpy().astype(np.float16) # [-1, 1] [3, h, w]
            incident_colored = norm_to_rgb(incident)
            incident_colored_hwc = chw2hwc(incident_colored)

            incident_colored_img = Image.fromarray(incident_colored_hwc)

            incident_pred = np.asarray(incident_colored_img) / 255.0 * 2.0 - 1.0

            return DiffCalibDepthOutput(
                incident_np=np.transpose(incident_pred, (2, 0, 1)),
                incident_colored=incident_colored_img,
                depth_np=depth_pred,
                depth_colored=depth_colored_img,
                uncertainty=pred_uncert,
            )

        elif mode == 'seg':
            # Clip output range
            depth_colored = depth_preds.clip(0, 255).cpu().numpy().astype(np.uint8)
            depth_colored_hwc = chw2hwc(depth_colored)
            depth_colored_img = Image.fromarray(depth_colored_hwc).resize(input_size)

            return DiffCalibSegOutput(
                seg_colored=depth_colored_img,
                uncertainty=None,
            )

        else:
            raise NotImplementedError


    def __encode_empty_text(self):
        """
        Encode text embedding for empty prompt
        """
        self.empty_text_embed = torch.load('empty_text_embed.pt').to(self.dtype)

    
    @torch.no_grad()
    def single_infer(
        self, 
        rgb_in: torch.Tensor, 
        num_inference_steps: int, 
        show_pbar: bool,
        mode: str = 'depth',
    ) -> torch.Tensor:
        """
        Perform an individual depth prediction without ensembling.

        Args:
            rgb_in (torch.Tensor):
                Input RGB image.
            num_inference_steps (int):
                Number of diffusion denoisign steps (DDIM) during inference.
            show_pbar (bool):
                Display a progress bar of diffusion denoising.

        Returns:
            torch.Tensor: Predicted depth map.
        """
        device = rgb_in.device

        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps  # [T]

        # Encode image
        # print('===1===')
        # import pdb
        # pdb.set_trace()
        rgb_latent = self.encode_rgb(rgb_in)

        # Initial incident map (noise)
        incident_latent = torch.randn(
            rgb_latent.shape, device=device, dtype=self.dtype)  # [B, 4, h, w]
        # Initial normal map (noise)
        # normal_latent = rgb_latent.clone()
        normal_latent = torch.randn(
            rgb_latent.shape, device=device, dtype=self.dtype)  # [B, 4, h, w]
        # pdb.set_trace()
        # Batched empty text embedding
        # if self.empty_text_embed is None:
        #     self.__encode_empty_text()
        batch_empty_text_embed = self.empty_text_embed.repeat(
            (rgb_latent.shape[0], 1, 1)
        )  # [B, 2, 1024]

        # Denoising loop
        if show_pbar:
            iterable = tqdm(
                enumerate(timesteps),
                total=len(timesteps),
                leave=False,
                desc=" " * 4 + "Diffusion denoising",
            )
        else:
            iterable = enumerate(timesteps)

        for i, t in iterable:
            unet_input = torch.cat(
                [rgb_latent, incident_latent, normal_latent], dim=1
            )  # this order is important

            # predict the noise residual
            # print('===2===')
            noise_pred = self.unet(
                unet_input, t, encoder_hidden_states=batch_empty_text_embed
            ).sample  # [B, 4, h, w]

            # compute the previous noisy sample x_t -> x_t-1
            noise_pred_incident = noise_pred[:, :4]
            noise_pred_normal = noise_pred[:, 4:]

            # compute the previous noisy sample x_t -> x_t-1
            # print('===3===')
            incident_latent = self.scheduler.step(noise_pred_incident, t, incident_latent).prev_sample
            normal_latent = self.scheduler.step(noise_pred_normal, t, normal_latent).prev_sample

        if mode == 'depth':
            # decoder incident
            incident_map = self.decode_normal(incident_latent)
            incident_map = torch.clip(incident_map, -1.0, 1.0)

            depth = self.decode_depth(normal_latent)
            # clip prediction
            depth = torch.clip(depth, -1.0, 1.0)
            # shift to [0, 1]
            depth = (depth + 1.0) / 2.0
            return incident_map, depth

        elif mode == 'seg':
            seg = self.decode_seg(depth_latent)
            # clip prediction
            seg = seg.mean(dim=0)
            seg = torch.clip(seg, -1.0, 1.0)
            # # shift to [0, 1]
            seg = (seg + 1.0) / 2.0 
            # # shift to [0, 255]
            seg = seg * 255

            return seg
        elif mode == 'normal':
            incident_map = self.decode_normal(incident_latent)
            normal_map = self.decode_normal(normal_latent)
            incident_map = torch.clip(incident_map, -1.0, 1.0)
            normal_map = torch.clip(normal_map, -1.0, 1.0)
            return incident_map, normal_map
        else:
            raise NotImplementedError


    def encode_rgb(self, rgb_in: torch.Tensor) -> torch.Tensor:
        """
        Encode RGB image into latent.

        Args:
            rgb_in (torch.Tensor):
                Input RGB image to be encoded.

        Returns:
            torch.Tensor: Image latent
        """
        # encode
        h = self.vae.encoder(rgb_in)
        moments = self.vae.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        # scale latent
        rgb_latent = mean * self.rgb_latent_scale_factor
        return rgb_latent

    
    def decode_depth(self, depth_latent: torch.Tensor) -> torch.Tensor:
        """
        Decode depth latent into depth map.

        Args:
            depth_latent (torch.Tensor):
                Depth latent to be decoded.

        Returns:
            torch.Tensor: Decoded depth map.
        """
        # scale latent
        depth_latent = depth_latent / self.depth_latent_scale_factor
        # decode
        z = self.vae.post_quant_conv(depth_latent)
        stacked = self.vae.decoder(z)

        # mean of output channels
        depth_mean = stacked.mean(dim=1, keepdim=True)
        return depth_mean


    def decode_seg(self, seg_latent: torch.Tensor) -> torch.Tensor:
        """
        Decode depth latent into depth map.

        Args:
            depth_latent (torch.Tensor):
                Depth latent to be decoded.

        Returns:
            torch.Tensor: Decoded depth map.
        """
        # scale latent
        seg_latent = seg_latent / self.seg_latent_scale_factor
        # decode
        z = self.vae.post_quant_conv(seg_latent)
        seg = self.vae.decoder(z)

        return seg
    
    def decode_normal(self, normal_latent: torch.Tensor) -> torch.Tensor:
        """
        Decode depth latent into depth map.

        Args:
            depth_latent (torch.Tensor):
                Depth latent to be decoded.

        Returns:
            torch.Tensor: Decoded depth map.
        """
        # scale latent
        normal_latent = normal_latent / self.seg_latent_scale_factor
        # decode
        z = self.vae.post_quant_conv(normal_latent)
        seg = self.vae.decoder(z)

        return seg
