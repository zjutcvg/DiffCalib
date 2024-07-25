import torch
from torch.utils.data import Dataset
from PIL import Image
import json
from torchvision import transforms
import os
import os.path as osp
import numpy as np

from marigold.util.image_util import ResizeLongestEdge

class DepthJsonDataset(Dataset):
    def __init__(self, json_path, dataset_root, processing_res=0):
        """
        Args:
            rgb_paths (list of str):
            transform (callable, optional): 
        """

        with open(json_path, 'r') as f:
            self.annos = json.load(f)['files']

        self.processing_res = processing_res
        self.dataset_root = dataset_root

        if self.processing_res != 0:
            self.transform = transforms.Compose( 
                [
                    ResizeLongestEdge(processing_res, interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.ToTensor(), # -> [0, 1]
                ]
            )
        else:
            self.transform = transforms.Compose( 
                [
                    # ResizeLongestEdge(processing_res, interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.ToTensor(), # -> [0, 1]
                ]
            )

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, idx):
        img_path = osp.join(self.dataset_root, self.annos[idx]['rgb'])
        depth_path = osp.join(self.dataset_root, self.annos[idx]['depth'])
        cam_in = torch.tensor(self.annos[idx]['cam_in']) # The cam_in should be the intrinsic of depth map

        # preprocess
        image = Image.open(img_path).convert('RGB')
        if 'kitti' in self.dataset_root:
            # kb crop
            height = image.size[1]
            width = image.size[0]
            top_margin = int(height - 352)
            left_margin = int((width - 1216) / 2)
            image = image.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))

            # cam_in[2] -= left_margin # cx
            # cam_in[3] -= top_margin # cy
            
        image_transformed = self.transform(image)
        
        # image_transformed.shape: [3, H, W]
        # image.size: [W, H]

        # resize_ratio = image_transformed.shape[2] / image.size[0]
        # cam_in *= resize_ratio # fx, fy, cx, cy

        output = dict(
            rgb=img_path,
            depth=depth_path,
            cam_in=cam_in,
            images=image_transformed,
        )

        if 'diode' in self.dataset_root:
            depth_mask_path =  osp.join(self.dataset_root, self.annos[idx]['depth_mask'])
            output['depth_mask'] = depth_mask_path
        
        return output