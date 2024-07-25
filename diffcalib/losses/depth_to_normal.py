import numpy as np
import torch
import torch.nn as nn

class Backprojection(nn.Module):
    """Layer to backproject a depth image given the camera intrinsics
    Attributes
        xy (Nx3x(HxW)): homogeneous pixel coordinates on regular grid
    """
    def __init__(self, height, width):
        """
        Args:
            height (int): image height
            width (int): image width
        """
        super(Backprojection, self).__init__()

        self.height = height
        self.width = width

        # generate regular grid
        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        id_coords = torch.tensor(id_coords, device="cuda")

        # generate homogeneous pixel coordinates
        # self.ones = nn.Parameter(torch.ones(1, 1, self.height * self.width),
        #                          requires_grad=False)
        ones = torch.ones(1, 1, self.height * self.width, device="cuda")
        xy = torch.unsqueeze(
            torch.stack([id_coords[0].view(-1), id_coords[1].view(-1)], 0),
            0
            )
        xy = torch.cat([xy, ones], 1)
        #self.xy = nn.Parameter(self.xy, requires_grad=False)
        self.register_buffer('xy', xy, persistent=False)
        self.register_buffer('ones', ones, persistent=False)

    def forward(self, depth, inv_K, img_like_out=False):
        """
        Args:
            depth (Nx1xHxW): depth map
            inv_K (Nx4x4): inverse camera intrinsics
            img_like_out (bool): if True, the output shape is Nx4xHxW; else Nx4x(HxW)
        Returns:
            points (Nx4x(HxW)): 3D points in homogeneous coordinates
        """
        depth = depth.contiguous()

        xy = self.xy.repeat(depth.shape[0], 1, 1)
        ones = self.ones.repeat(depth.shape[0],1,1)
        
        points = torch.matmul(inv_K[:, :3, :3], xy)
        points = depth.view(depth.shape[0], 1, -1) * points
        points = torch.cat([points, ones], 1)

        if img_like_out:
            points = points.reshape(depth.shape[0], 4, self.height, self.width)
        return points


def get_surface_normalv2(xyz, patch_size=5, mask_valid=None):
    """
    xyz: xyz coordinates, in [b, h, w, c]
    patch: [p1, p2, p3,
            p4, p5, p6,
            p7, p8, p9]
    surface_normal = [(p9-p1) x (p3-p7)] + [(p6-p4) - (p8-p2)]
    return: normal [h, w, 3, b]
    """
    b, h, w, c = xyz.shape
    half_patch = patch_size // 2

    if mask_valid == None:
        mask_valid = xyz[:, :, :, 2] > 0 # [b, h, w]
    mask_pad = torch.zeros((b, h + patch_size - 1, w + patch_size - 1), device=mask_valid.device).bool()
    mask_pad[:, half_patch:-half_patch, half_patch:-half_patch] = mask_valid
    
    xyz_pad = torch.zeros((b, h + patch_size - 1, w + patch_size - 1, c), dtype=xyz.dtype, device=xyz.device)
    xyz_pad[:, half_patch:-half_patch, half_patch:-half_patch, :] = xyz

    xyz_left = xyz_pad[:, half_patch:half_patch + h, :w, :]  # p4
    xyz_right = xyz_pad[:, half_patch:half_patch + h, -w:, :]  # p6
    xyz_top = xyz_pad[:, :h, half_patch:half_patch + w, :]  # p2
    xyz_bottom = xyz_pad[:, -h:, half_patch:half_patch + w, :]  # p8
    xyz_horizon = xyz_left - xyz_right  # p4p6
    xyz_vertical = xyz_top - xyz_bottom  # p2p8

    xyz_left_in = xyz_pad[:, half_patch:half_patch + h, 1:w+1, :]  # p4
    xyz_right_in = xyz_pad[:, half_patch:half_patch + h, patch_size-1:patch_size-1+w, :]  # p6
    xyz_top_in = xyz_pad[:, 1:h+1, half_patch:half_patch + w, :]  # p2
    xyz_bottom_in = xyz_pad[:, patch_size-1:patch_size-1+h, half_patch:half_patch + w, :]  # p8
    xyz_horizon_in = xyz_left_in - xyz_right_in  # p4p6
    xyz_vertical_in = xyz_top_in - xyz_bottom_in  # p2p8

    n_img_1 = torch.cross(xyz_horizon_in, xyz_vertical_in, dim=3)
    n_img_2 = torch.cross(xyz_horizon, xyz_vertical, dim=3)

    # re-orient normals consistently
    orient_mask = torch.sum(n_img_1 * xyz, dim=3) > 0
    n_img_1[orient_mask] *= -1
    orient_mask = torch.sum(n_img_2 * xyz, dim=3) > 0
    n_img_2[orient_mask] *= -1

    n_img1_L2 = torch.sqrt(torch.sum(n_img_1 ** 2, dim=3, keepdim=True))
    n_img1_norm = n_img_1 / (n_img1_L2 + 1e-8)

    n_img2_L2 = torch.sqrt(torch.sum(n_img_2 ** 2, dim=3, keepdim=True))
    n_img2_norm = n_img_2 / (n_img2_L2 + 1e-8)

    # average 2 norms
    n_img_aver = n_img1_norm + n_img2_norm
    n_img_aver_L2 = torch.sqrt(torch.sum(n_img_aver ** 2, dim=3, keepdim=True) + 1e-8)
    n_img_aver_norm = n_img_aver / (n_img_aver_L2 + 1e-8)
    # re-orient normals consistently
    orient_mask = torch.sum(n_img_aver_norm * xyz, dim=3) > 0
    n_img_aver_norm[orient_mask] *= -1
    #n_img_aver_norm_out = n_img_aver_norm.permute((1, 2, 3, 0))  # [h, w, c, b]

    # get mask for normals
    mask_p4p6 = mask_pad[:, half_patch:half_patch + h, :w] & mask_pad[:, half_patch:half_patch + h, -w:]
    mask_p2p8 = mask_pad[:, :h, half_patch:half_patch + w] & mask_pad[:, -h:, half_patch:half_patch + w]
    mask_normal = mask_p2p8 & mask_p4p6
    n_img_aver_norm[~mask_normal] = 0

    # a = torch.sum(n_img1_norm_out*n_img2_norm_out, dim=2).cpu().numpy().squeeze()
    # plt.imshow(np.abs(a), cmap='rainbow')
    # plt.show()
    return n_img_aver_norm.permute(0, 3, 1, 2).contiguous(), mask_normal[:, None, :, :] # [b, h, w, 3]

class Depth2Normal(nn.Module):
    """Layer to compute surface normal from depth map
    """
    def __init__(self,):
        """
        Args:
            height (int): image height
            width (int): image width
        """
        super(Depth2Normal, self).__init__()
    
    def init_img_coor(self, height, width):
        """
        Args:
            height (int): image height
            width (int): image width
        """
        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device="cuda"),
                               torch.arange(0, width, dtype=torch.float32, device="cuda")], indexing='ij')
        meshgrid = torch.stack((x, y))

        # # generate regular grid
        # meshgrid = np.meshgrid(range(width), range(height), indexing='xy')
        # id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        # id_coords = torch.tensor(id_coords)

        # generate homogeneous pixel coordinates
        ones = torch.ones((1, 1, height * width), device="cuda")
        # xy = torch.unsqueeze(
        #     torch.stack([id_coords[0].view(-1), id_coords[1].view(-1)], 0),
        #     0
        #     )
        xy = meshgrid.reshape(2, -1).unsqueeze(0)
        xy = torch.cat([xy, ones], 1)

        self.register_buffer('xy', xy, persistent=False)
        #self.register_buffer('ones', ones.cuda(), persistent=False)
    
    def back_projection(self, depth, inv_K, img_like_out=False):
        """
        Args:
            depth (Nx1xHxW): depth map
            inv_K (Nx4x4): inverse camera intrinsics
            img_like_out (bool): if True, the output shape is Nx4xHxW; else Nx4x(HxW)
        Returns:
            points (Nx4x(HxW)): 3D points in homogeneous coordinates
        """
        B, C, H, W = depth.shape
        depth = depth.contiguous()
        # xy = self.xy.repeat(depth.shape[0], 1, 1)
        xy = self.xy
        #ones = self.ones.repeat(depth.shape[0],1,1)
        
        points = torch.matmul(inv_K[:, :3, :3], xy)
        points = depth.view(depth.shape[0], 1, -1) * points
        #points = torch.cat([points, ones], 1)

        if img_like_out:
            points = points.reshape(depth.shape[0], 3, H, W)
        return points
    
    # def transfer_xyz(self, u0, v0, H, W, depth, focal_length):
    #     x_row = np.arange(0, W)
    #     x = np.tile(x_row, (H, 1))
    #     x = x.astype(np.float32)
    #     x = torch.from_numpy(x.copy()).cuda()
    #     u_m_u0 = x[None, None, :, :] - u0
    #     self.register_buffer('u_m_u0', u_m_u0, persistent=False)

    #     y_col = np.arange(0, H)  # y_col = np.arange(0, height)
    #     y = np.tile(y_col, (W, 1)).T
    #     y = y.astype(np.float32)
    #     y = torch.from_numpy(y.copy()).cuda()
    #     v_m_v0 = y[None, None, :, :] - v0
    #     self.register_buffer('v_m_v0', v_m_v0, persistent=False)

    #     pix_idx_mat = torch.arange(H*W).reshape((H, W)).cuda()
    #     self.register_buffer('pix_idx_mat', pix_idx_mat, persistent=False)

    #     x = self.u_m_u0 * depth / focal_length
    #     y = self.v_m_v0 * depth / focal_length
    #     z = depth
    #     pw = torch.cat([x, y, z], 1).permute(0, 2, 3, 1) # [b, h, w, c]
    #     return pw

    def forward(self, depth, intrinsics, masks):
        """
        Args:
            depth (Nx1xHxW): depth map
            #inv_K (Nx4x4): inverse camera intrinsics
            intrinsics (Nx4): camera intrinsics
        Returns:
            normal (Nx3xHxW): normalized surface normal
            mask (Nx1xHxW): valid mask for surface normal
        """
        B, C, H, W = depth.shape
        if 'xy' not in self._buffers:
            self.init_img_coor(height=H, width=W)
        # Compute 3D point cloud
        inv_K = intrinsics.inverse()
        xyz = self.back_projection(depth, inv_K) # [N, 4, HxW]
        xyz = xyz.view(depth.shape[0], 3, H, W)
        xyz = xyz[:,:3].permute(0, 2, 3, 1).contiguous() # [b, h, w, c]

        # focal_length = intrinsics[:, 0, 0][:, None, None, None]
        # u0 = intrinsics[:, 0, 2][:, None, None, None]
        # v0 = intrinsics[:, 1, 2][:, None, None, None]        
        # xyz2 = self.transfer_xyz(u0, v0, H, W, depth, focal_length)

        normals, normal_masks = get_surface_normalv2(xyz, mask_valid=masks.squeeze())
        normal_masks = normal_masks & masks
        return normals, normal_masks



if __name__ == '__main__':
    d2n = Depth2Normal()
    depth = np.random.randn(2, 1, 20, 22)
    intrin = np.array([[300, 0, 10], [0, 300, 10], [0,0,1]])
    intrinsics = np.stack([intrin, intrin], axis=0)

    depth_t = torch.from_numpy(depth).cuda().float()
    intrinsics = torch.from_numpy(intrinsics).cuda().float()
    normal = d2n(depth_t, intrinsics)
    normal2 = d2n(depth_t, intrinsics)
    print(normal)