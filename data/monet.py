import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class MonetDataset(Dataset):
    """Lazy load Monet dataset based on the provided data_path"""
    def __init__(self, data_path: str, center_crop: bool = False):
        image_names = os.listdir(data_path)
        self.image_paths = [
            os.path.join(data_path, image_name) \
            for image_name in image_names
        ]

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(256) if center_crop else transforms.RandomCrop(256),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx) -> np.array:
        im = Image.open(self.image_paths[idx])
        im = self.image_transforms(im)
        return im