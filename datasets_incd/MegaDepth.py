import os, io, glob, natsort, random
from einops import rearrange
import h5py
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from tools.tools import coords_gridN, resample_rgb, apply_augmentation

def intrinsic2incidence(K, b, h, w, device=None):
    coords = coords_gridN(b, h, w, device)

    x, y = torch.split(coords, 1, dim=1)
    x = (x + 1) / 2.0 * w
    y = (y + 1) / 2.0 * h
    # pdb.set_trace()
    pts3d = torch.cat([x, y, torch.ones_like(x)], dim=1)
    pts3d = rearrange(pts3d, 'b d h w -> b h w d')
    pts3d = pts3d.unsqueeze(dim=4)

    K_ex = K.view([b, 1, 1, 3, 3])
    pts3d = torch.linalg.inv(K_ex) @ pts3d
    pts3d = torch.nn.functional.normalize(pts3d, dim=3)

    # Assuming pts3d is a tensor with values in the range [-1, 1]
    # import pdb
    # pdb.set_trace()
    # normalized_pts3d = (pts3d.squeeze() + 1) / 2.0

    # Convert to a PIL Image
    # img_array = (normalized_pts3d.cpu().numpy() * 255).astype('uint8')
    # img = Image.fromarray(img_array)

    # return img
    return pts3d.squeeze().permute(2,0,1)
def coords_gridN(batch, ht, wd, device):
    if device is not None:
        coords = torch.meshgrid(
            (
                torch.linspace(-1 + 1 / ht, 1 - 1 / ht, ht, device=device),
                torch.linspace(-1 + 1 / wd, 1 - 1 / wd, wd, device=device),
            )
        )
    else:
        coords = torch.meshgrid(
            (
                torch.linspace(-1 + 1 / ht, 1 - 1 / ht, ht),
                torch.linspace(-1 + 1 / wd, 1 - 1 / wd, wd),
            )
        )
    coords = torch.stack((coords[1], coords[0]), dim=0)[
        None
    ].repeat(batch, 1, 1, 1)
    return coords

class MegaDepth:
    def __init__(self, data_root, ht=384, wt=512, shuffleseed=None, split='train') -> None:
        split_path = os.path.join('splits', 'megadepth_{}.txt'.format(split))

        with open(split_path) as file:
            data_names = [line.rstrip() for line in file]

        if split == 'train':
            if shuffleseed is not None:
                random.seed(shuffleseed)
            random.shuffle(data_names)

        self.data_root = data_root

        self.wt, self.ht = wt, ht
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.tensor = transforms.ToTensor()
        self.data_names = data_names

        self.datasetname = 'MegaDepth'

    def __len__(self):
        return len(self.data_names)

    def load_im(self, im_ref):
        im = Image.open(im_ref)
        return im

    def __getitem__(self, idx):
        # read intrinsics of original size
        scene_name, stem_name_color, stem_name_intrinsic = self.data_names[idx].split(' ')
        h5pypath = os.path.join(self.data_root, '{}.hdf5'.format(scene_name))

        with h5py.File(h5pypath, 'r') as hf:
            K_color = np.array(hf['intrinsic'][stem_name_intrinsic])

            # Load positive pair data
            rgb = self.load_im(io.BytesIO(np.array(hf['color'][stem_name_color])))
            
            raw = rgb.resize((1024, 1024), Image.BILINEAR)
            rgb_raw = torch.as_tensor(np.array(raw)).permute(2, 0, 1).contiguous()
            w, h = rgb.size
            
            # resize
            rgb = rgb.resize((self.wt, self.ht))

            scaleM = np.eye(3)
            scaleM[0, 0] = self.wt / w
            scaleM[1, 1] = self.ht / h
            aspect_ratio_restoration = (scaleM[1, 1] / scaleM[0, 0]).item()

            K = torch.from_numpy(scaleM @ K_color).float()

        # Recompute camera intrinsic matrix due to the resize
        rgb = self.normalize(self.tensor(rgb))

        # Save RAW
        K_raw = torch.clone(K)
        # rgb_raw = torch.clone(rgb)

        # K to incident field
        incidence_gt = intrinsic2incidence(K, 1, self.wt, self.ht, rgb.device)
        assert incidence_gt.min() >= -1 and incidence_gt.max() <= 1
        
        # Exportation
        data_dict = {
            'K': K,
            'pixel_values': rgb,
            'conditioning_pixel_values': incidence_gt,
            'normal_pixel_values': incidence_gt,
            'K_raw': K_raw,
            'rgb_raw': rgb_raw,
            'aspect_ratio_restoration': aspect_ratio_restoration,
            'datasetname': self.datasetname,
            # Only Required in Crop Evaluation
            'T': torch.eye(3, dtype=torch.float32),
            'scaleM': scaleM.astype(np.float32),
            'size_wo_change': (h, w),
        }

        return data_dict