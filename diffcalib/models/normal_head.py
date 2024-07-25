# This code is copied and modified from https://github.com/baegwangbin/surface_normal_uncertainty/blob/main/models/submodules/decoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F

########################################################################################################################


# Upsample + BatchNorm
class UpSampleBN(nn.Module):
    def __init__(self, skip_input, output_features):
        super(UpSampleBN, self).__init__()

        self._net = nn.Sequential(nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU(),
                                  nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU())

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        f = torch.cat([up_x, concat_with], dim=1)
        return self._net(f)


# Upsample + GroupNorm + Weight Standardization
class UpSampleGN(nn.Module):
    def __init__(self, skip_input, output_features):
        super(UpSampleGN, self).__init__()

        self._net = nn.Sequential(Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.GroupNorm(8, output_features),
                                  nn.LeakyReLU(),
                                  Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.GroupNorm(8, output_features),
                                  nn.LeakyReLU())

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        f = torch.cat([up_x, concat_with], dim=1)
        return self._net(f)


# Conv2d with weight standardization
class Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


# normalize
def norm_normalize(norm_out):
    min_kappa = 0.01
    norm_x, norm_y, norm_z, kappa = torch.split(norm_out, 1, dim=1)
    norm = torch.sqrt(norm_x ** 2.0 + norm_y ** 2.0 + norm_z ** 2.0) + 1e-10
    kappa = F.elu(kappa) + 1.0 + min_kappa
    final_out = torch.cat([norm_x / norm, norm_y / norm, norm_z / norm, kappa], dim=1)
    return final_out


# uncertainty-guided sampling (only used during training)
@torch.no_grad()
def sample_points(init_normal, gt_norm_mask, sampling_ratio, beta):
    device = init_normal.device
    B, _, H, W = init_normal.shape
    N = int(sampling_ratio * H * W)
    beta = beta

    # uncertainty map
    uncertainty_map = -1 * init_normal[:, 3, :, :]  # B, H, W

    # gt_invalid_mask (B, H, W)
    if gt_norm_mask is not None:
        gt_invalid_mask = F.interpolate(gt_norm_mask.float(), size=[H, W], mode='nearest')
        gt_invalid_mask = gt_invalid_mask[:, 0, :, :] < 0.5
        uncertainty_map[gt_invalid_mask] = -1e4

    # (B, H*W)
    _, idx = uncertainty_map.view(B, -1).sort(1, descending=True)

    # importance sampling
    if int(beta * N) > 0:
        importance = idx[:, :int(beta * N)]    # B, beta*N

        # remaining
        remaining = idx[:, int(beta * N):]     # B, H*W - beta*N

        # coverage
        num_coverage = N - int(beta * N)

        if num_coverage <= 0:
            samples = importance
        else:
            coverage_list = []
            for i in range(B):
                idx_c = torch.randperm(remaining.size()[1])    # shuffles "H*W - beta*N"
                coverage_list.append(remaining[i, :][idx_c[:num_coverage]].view(1, -1))     # 1, N-beta*N
            coverage = torch.cat(coverage_list, dim=0)                                      # B, N-beta*N
            samples = torch.cat((importance, coverage), dim=1)                              # B, N

    else:
        # remaining
        remaining = idx[:, :]  # B, H*W

        # coverage
        num_coverage = N

        coverage_list = []
        for i in range(B):
            idx_c = torch.randperm(remaining.size()[1])  # shuffles "H*W - beta*N"
            coverage_list.append(remaining[i, :][idx_c[:num_coverage]].view(1, -1))  # 1, N-beta*N
        coverage = torch.cat(coverage_list, dim=0)  # B, N-beta*N
        samples = coverage

    # point coordinates
    rows_int = samples // W         # 0 for first row, H-1 for last row
    rows_float = rows_int / float(H-1)         # 0 to 1.0
    rows_float = (rows_float * 2.0) - 1.0       # -1.0 to 1.0

    cols_int = samples % W          # 0 for first column, W-1 for last column
    cols_float = cols_int / float(W-1)         # 0 to 1.0
    cols_float = (cols_float * 2.0) - 1.0       # -1.0 to 1.0

    point_coords = torch.zeros(B, 1, N, 2)
    point_coords[:, 0, :, 0] = cols_float             # x coord
    point_coords[:, 0, :, 1] = rows_float             # y coord
    point_coords = point_coords.to(device)
    return point_coords, rows_int, cols_int

#############################################################################

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        # hyper-parameter for sampling
        self.sampling_ratio = 0.4
        self.importance_ratio = 0.7

        # # normal
        # self.conv_norm_2 = nn.Conv2d(self.feat_channels[3], self.feat_channels[3], kernel_size=1, stride=1, padding=0)
        # self.up_norm_1 = UpSampleBN(skip_input=self.feat_channels[3] + self.num_ch_dec[3], output_features=self.num_ch_dec[3])
        # self.up_norm_2 = UpSampleBN(skip_input=self.num_ch_dec[3] + self.num_ch_dec[2], output_features=self.num_ch_dec[2])
        # self.up_norm_3 = UpSampleBN(skip_input=self.num_ch_dec[2] + self.num_ch_dec[1], output_features=self.num_ch_dec[1])
        self.up_norm_4 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.num_ch_dec[1], self.num_ch_dec[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.num_ch_dec[0]),
            nn.LeakyReLU(),
            nn.Conv2d(self.num_ch_dec[0], self.num_ch_dec[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.num_ch_dec[0]),
            nn.LeakyReLU(),
        )

        # produces 1/8 res output
        self.out_conv_res8 = nn.Sequential(
            nn.Conv2d(self.num_ch_dec[2], self.num_ch_dec[0], kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(self.num_ch_dec[0], self.num_ch_dec[0], kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(self.num_ch_dec[0], self.num_ch_dec[0], kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(self.num_ch_dec[0], 4, kernel_size=3, stride=1, padding=1)
        )
        

        # produces 1/4 res output
        self.out_conv_res4 = nn.Sequential(
            nn.Conv1d(self.num_ch_dec[2] + 4, self.num_ch_dec[0], kernel_size=1), nn.ReLU(),
            nn.Conv1d(self.num_ch_dec[0], self.num_ch_dec[0], kernel_size=1), nn.ReLU(),
            nn.Conv1d(self.num_ch_dec[0], self.num_ch_dec[0], kernel_size=1), nn.ReLU(),
            nn.Conv1d(self.num_ch_dec[0], 4, kernel_size=1),
        )

        # produces 1/2 res output
        self.out_conv_res2 = nn.Sequential(
            nn.Conv1d(self.num_ch_dec[1] + 4, self.num_ch_dec[0], kernel_size=1), nn.ReLU(),
            nn.Conv1d(self.num_ch_dec[0], self.num_ch_dec[0], kernel_size=1), nn.ReLU(),
            nn.Conv1d(self.num_ch_dec[0], self.num_ch_dec[0], kernel_size=1), nn.ReLU(),
            nn.Conv1d(self.num_ch_dec[0], 4, kernel_size=1),
        )

        # produces 1/1 res output
        self.out_conv_res1 = nn.Sequential(
            nn.Conv1d(self.num_ch_dec[0] + 4, self.num_ch_dec[0], kernel_size=1), nn.ReLU(),
            nn.Conv1d(self.num_ch_dec[0], self.num_ch_dec[0], kernel_size=1), nn.ReLU(),
            nn.Conv1d(self.num_ch_dec[0], self.num_ch_dec[0], kernel_size=1), nn.ReLU(),
            nn.Conv1d(self.num_ch_dec[0], 4, kernel_size=1),
        )

    def forward(self, features_mono, training, norms=None, **kwargs):
        outputs = {}
        if training == True:
            assert norms is not None
            norms = norms[0]
            gt_norm_mask = (norms[:, 0:1, :, :] == 0) & (norms[:, 1:2, :, :] == 0) & (norms[:, 2:3, :, :] == 0)
            gt_norm_mask = ~gt_norm_mask

        x_d2 = x_8                      # x_d2:     1/8 res
        x_d3 = x_4                      # x_d3:     1/4 res
        x_d4 = self.up_norm_4(x_d3)     # x_d4:     1/2 res

        # 1/8 res output
        out_res8 = self.out_conv_res8(x_d2)             # out_res8: 1/8 res output
        out_res8 = norm_normalize(out_res8)             # out_res8: 1/8 res output

        ################################################################################################################
        # out_res4
        ################################################################################################################

        if training == True:
            # upsampling ... out_res8: [2, 4, 60, 80] -> out_res8_res4: [2, 4, 120, 160]
            out_res8_res4 = F.interpolate(out_res8, scale_factor=2, mode='bilinear', align_corners=True)
            B, _, H, W = out_res8_res4.shape

            # samples: [B, 1, N, 2]
            point_coords_res4, rows_int, cols_int = sample_points(out_res8_res4.detach(), gt_norm_mask,
                                                                  sampling_ratio=self.sampling_ratio,
                                                                  beta=self.importance_ratio)

            # output (needed for evaluation / visualization)
            out_res4 = out_res8_res4

            # grid_sample feature-map
            feat_res4 = F.grid_sample(x_d2, point_coords_res4, mode='bilinear', align_corners=True)  # (B, 512, 1, N)
            init_pred = F.grid_sample(out_res8, point_coords_res4, mode='bilinear', align_corners=True)  # (B, 4, 1, N)
            feat_res4 = torch.cat([feat_res4, init_pred], dim=1)  # (B, 512+4, 1, N)

            # prediction (needed to compute loss)
            samples_pred_res4 = self.out_conv_res4(feat_res4[:, :, 0, :])  # (B, 4, N)
            samples_pred_res4 = norm_normalize(samples_pred_res4)  # (B, 4, N) - normalized

            for i in range(B):
                out_res4[i, :, rows_int[i, :], cols_int[i, :]] = samples_pred_res4[i, :, :]

        else:
            # grid_sample feature-map
            feat_map = F.interpolate(x_d2, scale_factor=2, mode='bilinear', align_corners=True)
            init_pred = F.interpolate(out_res8, scale_factor=2, mode='bilinear', align_corners=True)
            feat_map = torch.cat([feat_map, init_pred], dim=1)  # (B, 512+4, H, W)
            B, _, H, W = feat_map.shape

            # try all pixels
            out_res4 = self.out_conv_res4(feat_map.view(B, self.num_ch_dec[2] + 4, -1))  # (B, 4, N)
            out_res4 = norm_normalize(out_res4)  # (B, 4, N) - normalized
            out_res4 = out_res4.view(B, 4, H, W)
            samples_pred_res4 = point_coords_res4 = None

        ################################################################################################################
        # out_res2
        ################################################################################################################

        if training == True:

            # upsampling ... out_res4: [2, 4, 120, 160] -> out_res4_res2: [2, 4, 240, 320]
            out_res4_res2 = F.interpolate(out_res4, scale_factor=2, mode='bilinear', align_corners=True)
            B, _, H, W = out_res4_res2.shape

            # samples: [B, 1, N, 2]
            point_coords_res2, rows_int, cols_int = sample_points(out_res4_res2.detach(), gt_norm_mask,
                                                                  sampling_ratio=self.sampling_ratio,
                                                                  beta=self.importance_ratio)

            # output (needed for evaluation / visualization)
            out_res2 = out_res4_res2

            # grid_sample feature-map
            feat_res2 = F.grid_sample(x_d3, point_coords_res2, mode='bilinear', align_corners=True)  # (B, 256, 1, N)
            init_pred = F.grid_sample(out_res4, point_coords_res2, mode='bilinear', align_corners=True)  # (B, 4, 1, N)
            feat_res2 = torch.cat([feat_res2, init_pred], dim=1)  # (B, 256+4, 1, N)

            # prediction (needed to compute loss)
            samples_pred_res2 = self.out_conv_res2(feat_res2[:, :, 0, :])  # (B, 4, N)
            samples_pred_res2 = norm_normalize(samples_pred_res2)  # (B, 4, N) - normalized

            for i in range(B):
                out_res2[i, :, rows_int[i, :], cols_int[i, :]] = samples_pred_res2[i, :, :]

        else:
            # grid_sample feature-map
            feat_map = F.interpolate(x_d3, scale_factor=2, mode='bilinear', align_corners=True)
            init_pred = F.interpolate(out_res4, scale_factor=2, mode='bilinear', align_corners=True)
            feat_map = torch.cat([feat_map, init_pred], dim=1)  # (B, 512+4, H, W)
            B, _, H, W = feat_map.shape

            out_res2 = self.out_conv_res2(feat_map.view(B, self.num_ch_dec[1] + 4, -1))  # (B, 4, N)
            out_res2 = norm_normalize(out_res2)  # (B, 4, N) - normalized
            out_res2 = out_res2.view(B, 4, H, W)
            samples_pred_res2 = point_coords_res2 = None

        ################################################################################################################
        # out_res1
        ################################################################################################################

        if training == True:
            # upsampling ... out_res4: [2, 4, 120, 160] -> out_res4_res2: [2, 4, 240, 320]
            out_res2_res1 = F.interpolate(out_res2, scale_factor=2, mode='bilinear', align_corners=True)
            B, _, H, W = out_res2_res1.shape

            # samples: [B, 1, N, 2]
            point_coords_res1, rows_int, cols_int = sample_points(out_res2_res1.detach(), gt_norm_mask,
                                                                  sampling_ratio=self.sampling_ratio,
                                                                  beta=self.importance_ratio)

            # output (needed for evaluation / visualization)
            out_res1 = out_res2_res1

            # grid_sample feature-map
            feat_res1 = F.grid_sample(x_d4, point_coords_res1, mode='bilinear', align_corners=True)  # (B, 128, 1, N)
            init_pred = F.grid_sample(out_res2, point_coords_res1, mode='bilinear', align_corners=True)  # (B, 4, 1, N)
            feat_res1 = torch.cat([feat_res1, init_pred], dim=1)  # (B, 128+4, 1, N)

            # prediction (needed to compute loss)
            samples_pred_res1 = self.out_conv_res1(feat_res1[:, :, 0, :])  # (B, 4, N)
            samples_pred_res1 = norm_normalize(samples_pred_res1)  # (B, 4, N) - normalized

            for i in range(B):
                out_res1[i, :, rows_int[i, :], cols_int[i, :]] = samples_pred_res1[i, :, :]

        else:
            # grid_sample feature-map
            feat_map = F.interpolate(x_d4, scale_factor=2, mode='bilinear', align_corners=True)
            init_pred = F.interpolate(out_res2, scale_factor=2, mode='bilinear', align_corners=True)
            feat_map = torch.cat([feat_map, init_pred], dim=1)  # (B, 512+4, H, W)
            B, _, H, W = feat_map.shape

            out_res1 = self.out_conv_res1(feat_map.view(B, self.num_ch_dec[0] + 4, -1))  # (B, 4, N)
            out_res1 = norm_normalize(out_res1)  # (B, 4, N) - normalized
            out_res1 = out_res1.view(B, 4, H, W)
            samples_pred_res1 = point_coords_res1 = None
        
        norm_out_list = [out_res8, out_res4, out_res2, out_res1]
        norm_pred_list = [out_res8, samples_pred_res4, samples_pred_res2, samples_pred_res1]
        norm_coord_list = [None, point_coords_res4, point_coords_res2, point_coords_res1]

        outputs=dict(
            norm_out_list=norm_out_list,
            norm_pred_list=norm_pred_list,
            norm_coord_list=norm_coord_list,
        )
        return outputs