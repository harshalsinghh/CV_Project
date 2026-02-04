import os
import cv2
import torch
from torch.utils.data import Dataset


class LowLightDataset(Dataset):
    def __init__(self, low_dir, high_dir, img_size=128):
        self.low_dir = low_dir
        self.high_dir = high_dir
        self.img_size = img_size

        low_files = set(os.listdir(low_dir))
        high_files = set(os.listdir(high_dir))

        # Keep only matching filenames
        candidates = sorted(list(low_files & high_files))

        self.images = []
        for name in candidates:
            low_path = os.path.join(low_dir, name)
            high_path = os.path.join(high_dir, name)

            if cv2.imread(low_path) is not None and cv2.imread(high_path) is not None:
                self.images.append(name)

        if len(self.images) == 0:
            raise RuntimeError("No valid image pairs found!")

        print(f"Using {len(self.images)} valid image pairs")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        name = self.images[idx]

        low_path = os.path.join(self.low_dir, name)
        high_path = os.path.join(self.high_dir, name)

        low_img = cv2.imread(low_path)
        high_img = cv2.imread(high_path)

        low_img = cv2.cvtColor(low_img, cv2.COLOR_BGR2RGB)
        high_img = cv2.cvtColor(high_img, cv2.COLOR_BGR2RGB)

        low_img = cv2.resize(low_img, (self.img_size, self.img_size))
        high_img = cv2.resize(high_img, (self.img_size, self.img_size))

        low_img = torch.from_numpy(low_img).permute(2, 0, 1).float() / 255.0
        high_img = torch.from_numpy(high_img).permute(2, 0, 1).float() / 255.0

        return low_img, high_img
