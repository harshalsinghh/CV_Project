import os
import cv2
import torch
from torch.utils.data import Dataset


class LowLightDataset(Dataset):
    def __init__(self, low_dir, high_dir, img_size=128):
        self.low_dir = low_dir
        self.high_dir = high_dir
        self.img_size = img_size

        # list all image filenames from low-light folder
        self.images = sorted(os.listdir(low_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        low_path = os.path.join(self.low_dir, self.images[idx])
        high_path = os.path.join(self.high_dir, self.images[idx])

        low_img = cv2.imread(low_path)
        high_img = cv2.imread(high_path)

        if low_img is None or high_img is None:
            raise RuntimeError(f"Failed to load image pair: {self.images[idx]}")

        # BGR → RGB
        low_img = cv2.cvtColor(low_img, cv2.COLOR_BGR2RGB)
        high_img = cv2.cvtColor(high_img, cv2.COLOR_BGR2RGB)

        # Resize
        low_img = cv2.resize(low_img, (self.img_size, self.img_size))
        high_img = cv2.resize(high_img, (self.img_size, self.img_size))

        # Convert to tensor [0,1]
        low_img = torch.from_numpy(low_img).permute(2, 0, 1).float() / 255.0
        high_img = torch.from_numpy(high_img).permute(2, 0, 1).float() / 255.0

        return low_img, high_img
