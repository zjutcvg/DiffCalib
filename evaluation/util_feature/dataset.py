import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class RGBPathDataset(Dataset):
    def __init__(self, rgb_paths, transform=None):
        """
        Args:
            rgb_paths (list of str):
            transform (callable, optional): 
        """
        self.rgb_paths = rgb_paths
        self.transform = transform

        if self.transform is None:
            self.transform = transforms.Compose( 
                [
                    transforms.ToTensor(), # -> [0, 1]
                ]
            )

    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, idx):
        img_path = self.rgb_paths[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        output = dict(
            rgb=img_path,
            images=image,
        )
        return output