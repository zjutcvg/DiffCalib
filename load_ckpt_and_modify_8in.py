import pdb
import torch
import torch.nn as nn
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
model_id = "stabilityai/stable-diffusion-2-1"
pipe = StableDiffusionPipeline.from_pretrained('/mnt/nas/share/home/xugk/hxk/code/diffcalib_rep/checkpoint/AI-ModelScope/stable-diffusion-2-1')
# pipe = StableDiffusionPipeline.from_pretrained(model_id)

# input
unet_conv_in = pipe.unet.conv_in # Conv2d(4, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

new_conv_in = nn.Conv2d(
    in_channels=8,
    out_channels=unet_conv_in.out_channels, 
    kernel_size=unet_conv_in.kernel_size, 
    stride=unet_conv_in.stride, 
    padding=unet_conv_in.padding, 
    bias=True,
)

with torch.no_grad():
    new_conv_in.weight = nn.Parameter((unet_conv_in.weight.repeat(1, 2, 1, 1) / 2))
    new_conv_in.bias = torch.nn.Parameter(unet_conv_in.bias.clone())
    # new_conv_in.bias = unet_conv_in.bias.clone()

pipe.unet.conv_in = new_conv_in

pipe.save_pretrained("checkpoint/stable-diffusion-2-1-marigold-8inchannels")