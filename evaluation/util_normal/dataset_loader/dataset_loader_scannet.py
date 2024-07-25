import torch
from torchvision import transforms
from torch.utils.data.dataset import Dataset  # For custom datasets
import skimage.io as sio
import pickle
import numpy as np
import cv2
import os
from PIL import Image

from marigold.util.image_util import ResizeLongestEdge

class ScannetDataset(Dataset):
    def __init__(self, root='/mars/mnt/dgx/FrameNet/scannet-frames/',
                       usage='test',
                       train_test_split='./data/scannet_standard_train_test_val_split.pkl',
                       processing_res=0):
        
        # import pdb
        # pdb.set_trace()
        # Transforms
        self.root = root
        self.to_tensor = transforms.ToTensor()

        if processing_res != 0:
            self.transform = transforms.Compose( 
                [
                    ResizeLongestEdge(processing_res, interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.ToTensor(), # -> [0, 1]
                ]
            )
        else:
            self.transform = None

        self.train_test_plit = train_test_split
        self.data_info = pickle.load(open(train_test_split, 'rb'))[usage]

        self.idx = [i for i in range(0, len(self.data_info[0]), 1)]

        if usage == 'test':
            self.idx = [i for i in range(0, len(self.data_info[0]), 200)]

        self.data_len = len(self.idx)
        self.root = root

    def __getitem__(self, index):
        if 'framenet_train_test_split.pkl' in self.train_test_plit: # get proper path from framenet pkl
            color_filename = self.data_info[0][self.idx[index]]
            orient_filename = self.data_info[1][self.idx[index]][:-10] + 'normal.png'
            mask_filename = self.data_info[2][self.idx[index]]

            color_info = self.root + '/' + color_filename[27:]
            orient_info = self.root + '/' + orient_filename[27:]
            mask_info = self.root + '/' + mask_filename[27:]

        else:
            color_filename = self.data_info[0][self.idx[index]]
            orient_filename = self.data_info[1][self.idx[index]][:-10] + 'normal.png'
            mask_filename = self.data_info[2][self.idx[index]]

            color_info = self.root + '/' + color_filename
            orient_info = self.root + '/' + orient_filename
            mask_info = self.root + '/' + mask_filename

        # Image resize and load
        # color_img = cv2.resize(sio.imread(color_info), (320, 240), interpolation=cv2.INTER_CUBIC)
        orient_mask_tensor = cv2.resize(sio.imread(mask_info), (320, 240), interpolation=cv2.INTER_NEAREST)
        orient_img = cv2.resize(sio.imread(orient_info), (320, 240), interpolation=cv2.INTER_NEAREST)
        
        # color_img = sio.imread(color_info)
        # orient_mask_tensor = sio.imread(mask_info)
        # orient_img = sio.imread(orient_info)

        # To tensor
        orient_mask_tensor = torch.Tensor(orient_mask_tensor/255.0)
        Z = -self.to_tensor(orient_img) + 0.5 # NOTE: range from [-0.5, 0.5]
        Z = self.postprocess_normal(Z) # NOTE: range from [-1, 1]

        # input_tensor = np.zeros((3, color_img.shape[0], color_img.shape[1]), dtype='float32')
        # input_tensor[0:3, :, :] = color_tensor

        color_img = Image.open(color_info).convert("RGB")
        if self.transform:
            input_tensor = self.transform(color_img)

        return {'images': input_tensor, 'mask': orient_mask_tensor, 'Z': Z, 'rgb': color_filename}

    def __len__(self):
        return self.data_len

    def postprocess_normal(self, Z):
        Z = Z * 2
        Z[0] = -Z[0]
        return Z


# class Rectified2DOF(Dataset):
#     def __init__(self, root='/mars/mnt/dgx/FrameNet/scannet-frames/',
#                        usage='test',
#                        train_test_split='./data/rectified_2dofa_scannet.pkl'):
#         # Transforms
#         self.root = root
#         self.to_tensor = transforms.ToTensor()
#         if processing_res != 0:
#             self.transform = transforms.Compose( 
#                 [
#                     ResizeLongestEdge(processing_res, interpolation=transforms.InterpolationMode.BILINEAR),
#                     transforms.ToTensor(), # -> [0, 1]
#                 ]
#             )
#         else:
#             self.transform = None

#         self.train_test_plit = train_test_split

#         self.data_info = pickle.load(open(train_test_split, 'rb'))[usage]

#         self.idx_e2 = [i for i in range(0, len(self.data_info['e2']), 1)]
#         self.idx_me2 = [i for i in range(0, len(self.data_info['-e2']), 1)]

#         if usage == 'test':
#             self.idx_e2 = [i for i in range(0, len(self.data_info['e2']), 50)]
#             self.idx_me2 = [i for i in range(0, len(self.data_info['-e2']), 50)]

#         self.data_len = max((len(self.idx_e2), len(self.idx_me2)))
#         print('idx_e2: %d, idx_me2: %d' % (len(self.idx_e2), len(self.idx_me2)))

#         self.root = root

#     def __getitem__(self, index):
#         if np.random.ranf() < 2./3:
#             data_idx = self.idx_e2[index % len(self.idx_e2)]
#             data_split = 'e2'
#         else:
#             data_idx = self.idx_me2[index % len(self.idx_me2)]
#             data_split = '-e2'

#         color_info = os.path.join(self.root, self.data_info[data_split][data_idx])
#         mask_info = color_info.replace('color', 'orient-mask')
#         orient_mask_tensor = cv2.resize(sio.imread(mask_info), (320, 240), interpolation=cv2.INTER_NEAREST)
#         # orient_mask_tensor = sio.imread(mask_info)
#         mask_valid_size = np.sum((orient_mask_tensor > 0))

#         while mask_valid_size < 3e4:
#             data_split = 'e2'
#             index = np.random.randint(0, len(self.idx_e2))
#             data_idx = self.idx_e2[index % len(self.idx_e2)]
#             color_info = os.path.join(self.root, self.data_info[data_split][data_idx])
#             mask_info = color_info.replace('color', 'orient-mask')
#             orient_mask_tensor = cv2.resize(sio.imread(mask_info), (320, 240), interpolation=cv2.INTER_NEAREST)
#             # orient_mask_tensor = sio.imread(mask_info)
#             mask_valid_size = np.sum((orient_mask_tensor > 0))

#         orient_info = color_info.replace('color', 'normal')
#         gravity_info = color_info.replace('color.png', 'gravity.txt')
#         gravity_info = gravity_info.replace('scannet-frames', 'scannet-small-frames')

#         if data_split == 'e2':
#             aligned_directions = torch.tensor([0., 1., 0.], dtype=torch.float)
#             gravity_tensor = torch.tensor(np.loadtxt(gravity_info, dtype=np.float), dtype=torch.float)
#         elif data_split == '-e2':
#             aligned_directions = torch.tensor([0., 1., 0.], dtype=torch.float)
#             gravity_tensor = -torch.tensor(np.loadtxt(gravity_info, dtype=np.float), dtype=torch.float)

#         # Image resize and load
#         # color_img = cv2.resize(sio.imread(color_info), (320, 240), interpolation=cv2.INTER_CUBIC)
#         orient_img = cv2.resize(sio.imread(orient_info), (320, 240), interpolation=cv2.INTER_NEAREST)
        
#         # color_img = sio.imread(color_info)
#         # orient_img = sio.imread(orient_info)

#         # To tensor
#         # color_tensor = self.to_tensor(color_img)
#         orient_mask_tensor = torch.Tensor(orient_mask_tensor/255.0)
#         Z = -self.to_tensor(orient_img) + 0.5

#         # input_tensor = np.zeros((3, color_img.shape[0], color_img.shape[1]), dtype='float32')
#         # input_tensor[0:3, :, :] = color_tensor

#         color_img = Image.open(color_info).convert("RGB")
#         if self.transform:
#             input_tensor = self.transform(color_img)

#         return {'images': input_tensor, 'mask': orient_mask_tensor, 'Z': Z,
#                 'gravity': gravity_tensor, 'aligned_directions': aligned_directions, 'ga_split': data_split, 'rgb': color_info}

#     def __len__(self):
#         return self.data_len


# class Full2DOF(Dataset):
#     def __init__(self, root='/mars/mnt/dgx/FrameNet/scannet-frames/',
#                        usage='test',
#                        train_test_split='./data/full_2dofa_scannet.pkl'):
#         # Transforms
#         self.root = root
#         self.to_tensor = transforms.ToTensor()
#         if processing_res != 0:
#             self.transform = transforms.Compose( 
#                 [
#                     ResizeLongestEdge(processing_res, interpolation=transforms.InterpolationMode.BILINEAR),
#                     transforms.ToTensor(), # -> [0, 1]
#                 ]
#             )
#         else:
#             self.transform = None

#         self.train_test_plit = train_test_split

#         self.data_info = pickle.load(open(train_test_split, 'rb'))[usage]
#         self.data_with_ga = self.data_info['with_ga']
#         self.data_with_ga_e2 = self.data_with_ga['e2']
#         self.data_with_ga_me2 = self.data_with_ga['-e2']
#         self.data_no_ga = self.data_info['no_ga']

#         if usage == 'test':
#             self.data_with_ga_e2 = [self.data_with_ga['e2'][i] for i in range(0, len(self.data_with_ga['e2']), 50)]
#             self.data_with_ga_me2 = [self.data_with_ga['-e2'][i] for i in range(0, len(self.data_with_ga['-e2']), 50)]
#             self.data_no_ga = [self.data_info['no_ga'][i] for i in range(0, len(self.data_info['no_ga']), 50)]

#         self.ga_part_len = len(self.data_with_ga_e2) + len(self.data_with_ga_me2)
#         self.data_len = len(self.data_with_ga_e2) + len(self.data_with_ga_me2) + len(self.data_no_ga)

#         # index
#         self.idx_with_ga_e2 = [i for i in range(0, len(self.data_with_ga_e2), 1)]
#         self.idx_with_ga_me2 = [i for i in range(0, len(self.data_with_ga_me2), 1)]
#         self.idx_no_ga = [i for i in range(0, len(self.data_no_ga), 1)]

#         self.root = root

#     def __getitem__(self, index):
#         if np.random.ranf() < self.ga_part_len/float(self.data_len):
#             # draw from ga_part (e2 or -e2)
#             if np.random.ranf() < 2. / 3:
#                 data_idx = self.idx_with_ga_e2[index % len(self.idx_with_ga_e2)]
#                 data_split = 'e2'
#             else:
#                 data_idx = self.idx_with_ga_me2[index % len(self.idx_with_ga_me2)]
#                 data_split = '-e2'

#             color_info = os.path.join(self.root, self.data_with_ga[data_split][data_idx])
#         else:
#             # draw from no_ga_part
#             data_idx = self.idx_no_ga[index % len(self.idx_no_ga)]
#             data_split = 'no_ga'
#             color_info = os.path.join(self.root, self.data_no_ga[data_idx])

#         mask_info = color_info.replace('color', 'orient-mask')
#         orient_mask_tensor = cv2.resize(sio.imread(mask_info), (320, 240), interpolation=cv2.INTER_NEAREST)
#         # orient_mask_tensor = sio.imread(mask_info)
#         mask_valid_size = np.sum((orient_mask_tensor > 0))

#         while mask_valid_size < 3e4:
#             data_split = 'e2'
#             index = np.random.randint(0, len(self.idx_with_ga_e2))
#             data_idx = self.idx_with_ga_e2[index % len(self.idx_with_ga_e2)]
#             color_info = os.path.join(self.root, self.data_with_ga[data_split][data_idx])
#             mask_info = color_info.replace('color', 'orient-mask')
#             orient_mask_tensor = cv2.resize(sio.imread(mask_info), (320, 240), interpolation=cv2.INTER_NEAREST)
#             # orient_mask_tensor = sio.imread(mask_info)
#             mask_valid_size = np.sum((orient_mask_tensor > 0))

#         orient_info = color_info.replace('color', 'normal')
#         gravity_info = color_info.replace('color.png', 'gravity.txt')
#         gravity_info = gravity_info.replace('scannet-frames', 'scannet-small-frames')

#         aligned_directions, gravity_tensor = None, None

#         if data_split == 'e2':
#             aligned_directions = torch.tensor([0., 1., 0.], dtype=torch.float)
#             gravity_tensor = torch.tensor(np.loadtxt(gravity_info, dtype=np.float), dtype=torch.float)
#         elif data_split == '-e2':
#             aligned_directions = torch.tensor([0., 1., 0.], dtype=torch.float)
#             gravity_tensor = -torch.tensor(np.loadtxt(gravity_info, dtype=np.float), dtype=torch.float)
#         else:
#             # dummy values
#             aligned_directions = torch.tensor([0., 0., 0.], dtype=torch.float)
#             gravity_tensor = torch.tensor([0., 0., 0.], dtype=torch.float)

#         # Image resize and load
#         # color_img = cv2.resize(sio.imread(color_info), (320, 240), interpolation=cv2.INTER_CUBIC)
#         orient_img = cv2.resize(sio.imread(orient_info), (320, 240), interpolation=cv2.INTER_NEAREST)
#         # color_img = sio.imread(color_info)
#         # orient_img = sio.imread(orient_info)

#         # To tensor
#         # color_tensor = self.to_tensor(color_img)
#         orient_mask_tensor = torch.Tensor(orient_mask_tensor/255.0)
#         Z = -self.to_tensor(orient_img) + 0.5

#         # input_tensor = np.zeros((3, color_img.shape[0], color_img.shape[1]), dtype='float32')
#         # input_tensor[0:3, :, :] = color_tensor

#         color_img = Image.open(color_info).convert("RGB")
#         if self.transform:
#             input_tensor = self.transform(color_img)

#         return {'images': input_tensor, 'mask': orient_mask_tensor, 'Z': Z,
#                 'gravity': gravity_tensor, 'aligned_directions': aligned_directions, 'ga_split': data_split, 'rgb': color_info}

#     def __len__(self):
#         return self.data_len