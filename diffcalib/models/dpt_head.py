import torch
import torch.nn as nn

from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config

def _make_fusion_block(features, use_bn, size = None, fuse_flag=False):
    return FeatureFusionBlock_custom(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
        fuse_flag=fuse_flag,
    )

class ResidualConvUnit_custom(nn.Module):
    """Residual convolution module.
    """

    def __init__(self, features, activation, bn):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn

        self.groups=1

        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups
        )
        
        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups
        )

        if self.bn==True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """
        
        out = self.activation(x)
        out = self.conv1(out)
        if self.bn==True:
            out = self.bn1(out)
       
        out = self.activation(out)
        out = self.conv2(out)
        if self.bn==True:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)

class FeatureFusionBlock_custom(nn.Module):
    """Feature fusion block.
    """

    def __init__(self, features, activation, deconv=False, bn=False, expand=False, align_corners=True, size=None, fuse_flag=False):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock_custom, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups=1

        self.expand = expand
        out_features = features
        if self.expand==True:
            out_features = features//2
        
        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)

        if fuse_flag:
            self.resConfUnit1 = ResidualConvUnit_custom(features, activation, bn)
            
        self.resConfUnit2 = ResidualConvUnit_custom(features, activation, bn)
        
        self.skip_add = nn.quantized.FloatFunctional()

        self.size=size

    def forward(self, *xs, size=None):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)
            # output += res

        output = self.resConfUnit2(output)

        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        output = nn.functional.interpolate(
            output, **modifier, mode="bilinear", align_corners=self.align_corners
        )

        output = self.out_conv(output)

        return output

class Interpolate(nn.Module):
    """Interpolation module.
    """

    def __init__(self, scale_factor, mode, align_corners=False):
        """Init.

        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: interpolated data
        """

        x = self.interp(
            x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners
        )

        return x
        
class DPTHead(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, in_channels, upsample_flag=False):
        super(DPTHead, self).__init__()

        use_bn = False
        features = 256
        head_features_1 = features
        head_features_2 = 32
        fuse_flag = False
        non_negative = True

        self.input_conv = nn.Conv2d(in_channels, features, kernel_size=3, stride=1, padding=1)

        self.refinenet4 = _make_fusion_block(features, use_bn, fuse_flag=fuse_flag)
        self.refinenet3 = _make_fusion_block(features, use_bn, fuse_flag=fuse_flag)
        self.refinenet2 = _make_fusion_block(features, use_bn, fuse_flag=fuse_flag)
        self.refinenet1 = _make_fusion_block(features, use_bn, fuse_flag=fuse_flag)
        self.output_conv = nn.Sequential(
            nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1),
            # Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
            nn.ELU(True) if non_negative else nn.Identity(),
            # nn.Identity(),
        )

        self.in_channels = in_channels
        self.number_layers = 4
        self.upsample_flag = upsample_flag
        self.non_negative = non_negative
        
    
    def forward(self, features_mono):

        features_mono = self.input_conv(features_mono)

        if self.upsample_flag:
            features_mono = self.refinenet4(features_mono, size=(features_mono.shape[2] * 2, features_mono.shape[3] * 2))
            features_mono = self.refinenet3(features_mono, size=(features_mono.shape[2] * 2, features_mono.shape[3] * 2))
            features_mono = self.refinenet2(features_mono, size=(features_mono.shape[2] * 2, features_mono.shape[3] * 2))
            features_mono = self.refinenet1(features_mono, size=features_mono.shape[2:])
        else:
            features_mono = self.refinenet4(features_mono, size=features_mono.shape[2:])
            features_mono = self.refinenet3(features_mono, size=features_mono.shape[2:])
            features_mono = self.refinenet2(features_mono, size=features_mono.shape[2:])
            features_mono = self.refinenet1(features_mono, size=features_mono.shape[2:])

        out = self.output_conv(features_mono)

        if self.non_negative:
            out = out + 1

        # assert torch.all(out >= 0), print(out.min())
        out.squeeze(dim=1)

        return out
