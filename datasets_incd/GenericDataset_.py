import os, io, glob, natsort, random, copy
from einops import rearrange
import h5py
import torch
import numpy as np
import hashlib
from PIL import Image
from torchvision import transforms
from tools.tools import coords_gridN, resample_rgb, apply_augmentation
from torchvision.transforms import ColorJitter

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

def add_white_noise(rgb):
    w, h = rgb.size
    rgb = np.array(rgb).astype(np.float32)
    rgb = rgb + np.random.randint(9, size=(h, w, 3)) - 2
    rgb = np.clip(rgb, a_min=0, a_max=255)
    rgb = np.round(rgb).astype(np.uint8)
    rgb = Image.fromarray(rgb)
    return rgb

class GenericDataset:
    def __init__(self,
                 data_root,
                 ht=384, wt=512,
                 augmentation=False,
                 shuffleseed=None,
                 split='train',
                 datasetname='MegaDepth',
                 augscale=2.0,
                 no_change_prob=0.1,
                 coloraugmentation=False,
                 coloraugmentation_scale=0.1,
                 transformcategory='transform_calibration'
                 ) -> None:

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
        split_path = os.path.join('splits', '{}_{}.txt'.format(name_mapping[datasetname], split))
        print(split_path)
        with open(split_path) as file:
            data_names = [line.rstrip() for line in file]

        if split == 'train':
            if shuffleseed is not None:
                random.seed(shuffleseed)
            random.shuffle(data_names)
            self.training = True
        else:
            self.training = False

        self.data_root = data_root

        self.wt, self.ht = wt, ht
        self.normalize = transforms.Normalize(0.5, 0.5)
        self.tensor = transforms.ToTensor()

        self.data_names = data_names
        self.augmentation = augmentation
        self.augscale = augscale
        self.no_change_prob = no_change_prob
        self.coloraugmentation = coloraugmentation
        self.coloraugmentation_scale = coloraugmentation_scale

        self.datasetname = datasetname
        self.transformcategory = transformcategory

    def __len__(self):
        return len(self.data_names)

    def load_im(self, im_ref):
        im = Image.open(im_ref)
        return im

    def color_augmentation_fun(self, rgb):
        if random.uniform(0, 1) > 0.5:
            colorjitter = ColorJitter(
                brightness=self.coloraugmentation_scale,
                contrast=self.coloraugmentation_scale,
                saturation=self.coloraugmentation_scale,
                hue=self.coloraugmentation_scale / 3.14
            )
            rgb = colorjitter(rgb)
        rgb = add_white_noise(rgb)
        return rgb

    def __getitem__(self, idx):
        # While augmenting novel intrinsic, we follow:
        # Step 1 : Resize from original resolution to 480 x 640 (input height x input width)
        # Step 2 : Apply random spatial augmentation

        scene_name, stem_name = self.data_names[idx].split(' ')
        h5pypath = os.path.join(self.data_root, '{}.hdf5'.format(scene_name))

        if not os.path.exists(h5pypath):
            print("H5 file %s missing" % h5pypath)
            assert os.path.exists(h5pypath)

        with h5py.File(h5pypath, 'r') as hf:
            # Load Intrinsic
            K_color = np.array(hf['intrinsic'][stem_name])

            # Load RGB
            rgb = self.load_im(io.BytesIO(np.array(hf['color'][stem_name])))

        w, h = rgb.size

        # Step 1 method 1 : Resize
        # rgb = rgb.resize((self.wt, self.ht))

        # scaleM = np.eye(3)
        # scaleM[0, 0] = self.wt / w
        # scaleM[1, 1] = self.ht / h
        # aspect_ratio_restoration = (scaleM[1, 1] / scaleM[0, 0]).item()

        # K = torch.from_numpy(scaleM @ K_color).float()

        # Step 1 method 2: Resize and crop
        ratio = self.wt / min(w, h)
        w_, h_ = int(w * ratio + 0.5), int(h * ratio + 0.5)
        rgb = rgb.resize((w_, h_))

        delta_h = h_ - self.ht
        delta_w = w_ - self.wt

        top = delta_h // 2
        left = delta_w // 2
        rgb = rgb.crop((left, top, left + self.wt, top + self.ht))

        assert rgb.size == (768, 768)

        scaleM = np.eye(3)
        scaleM[0, 0] = ratio
        scaleM[1, 1] = ratio
        aspect_ratio_restoration = (scaleM[1, 1] / scaleM[0, 0]).item()

        K = torch.from_numpy(scaleM @ K_color).float()

        # adjust for crop
        K[0,2] -= left 
        K[1,2] -= top 

        # Color Augmentation only in training
        rgb = self.color_augmentation_fun(rgb) if self.coloraugmentation else rgb

        # Normalization
        rgb = self.normalize(self.tensor(rgb))

        # Save RAW
        K_raw = torch.clone(K)
        rgb_raw = torch.clone(rgb)

        # Step 2 : Random spatial augmentation
        # if self.augmentation:
        #     if self.training:
        #         rgb, K, T = apply_augmentation(
        #             rgb, K, seed=None, augscale=self.augscale, no_change_prob=self.no_change_prob
        #         )
        #     else:
        #         T = np.array(h5py.File(h5pypath, 'r')[self.transformcategory][stem_name])
        #         T = torch.from_numpy(T).float()
        #         K = torch.inverse(T) @ K

        #         _, h_, w_ = rgb.shape
        #         rgb = resample_rgb(rgb.unsqueeze(0), T, 1, h_, w_, rgb.device).squeeze(0)
        # else:
        T = torch.eye(3, dtype=torch.float32)
        
        # K to incident field
        incidence_gt = intrinsic2incidence(K, 1, self.wt, self.ht, rgb.device)
        assert incidence_gt.min() >= -1 and incidence_gt.max() <= 1
        
        # Exportation
        data_dict = {
            'K': K,
            'pixel_values': rgb,
            'conditioning_pixel_values': incidence_gt,
            'K_raw': K_raw,
            'rgb_raw': rgb_raw,
            'aspect_ratio_restoration': aspect_ratio_restoration,
            'datasetname': self.datasetname,
            # Only Required in Crop Evaluation
            'T': T,
            'scaleM': scaleM.astype(np.float32),
            'size_wo_change': (h, w),
        }

        return data_dict