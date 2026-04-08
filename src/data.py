import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class LowLightDataset(Dataset):
    def __init__(self, low_dir, high_dir, img_size=256):
        self.img_size = img_size
        self.pairs = []
        low_files = sorted(os.listdir(low_dir))
        for f in low_files:
            lp = os.path.join(low_dir, f)
            hp = os.path.join(high_dir, f)
            if os.path.exists(lp) and os.path.exists(hp):
                self.pairs.append((lp, hp))
        print(f"Using {len(self.pairs)} valid image pairs")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        lp, hp = self.pairs[idx]
        low  = cv2.cvtColor(cv2.imread(lp), cv2.COLOR_BGR2RGB)
        high = cv2.cvtColor(cv2.imread(hp), cv2.COLOR_BGR2RGB)
        low  = cv2.resize(low,  (self.img_size, self.img_size))
        high = cv2.resize(high, (self.img_size, self.img_size))
        low  = torch.from_numpy(low).permute(2,0,1).float() / 255.0
        high = torch.from_numpy(high).permute(2,0,1).float() / 255.0
        return low, high


class SyntheticLowLightDataset(Dataset):
    """
    Takes any normal-light image folder and synthesizes
    dark versions on the fly for pre-training.
    """
    def __init__(self, image_dir, img_size=256):
        self.img_size = img_size
        self.files = [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        print(f"Synthetic dataset: {len(self.files)} source images")

    def __len__(self):
        return len(self.files)

    def synthesize_dark(self, img):
        """Physics-inspired degradation pipeline."""
        img = img.numpy().astype(np.float32)

        # 1. Gamma reduction (underexposure)
        gamma = np.random.uniform(0.2, 0.5)
        dark = np.power(img, 1.0 / gamma)
        dark = np.clip(dark, 0, 1)

        # 2. Poisson noise (dominant in low light)
        dark_scaled = (dark * 255).astype(np.float32)
        noisy = np.random.poisson(dark_scaled) / 255.0
        noisy = np.clip(noisy, 0, 1).astype(np.float32)

        # 3. Color shift (white balance drift)
        shift = np.random.uniform(0.85, 1.15, (3, 1, 1)).astype(np.float32)
        noisy = np.clip(noisy * shift, 0, 1)

        # 4. Slight blur (lens/motion in dark)
        noisy_hwc = noisy.transpose(1, 2, 0)
        if np.random.rand() > 0.5:
            noisy_hwc = cv2.GaussianBlur(noisy_hwc, (3, 3), 0.5)
        noisy = noisy_hwc.transpose(2, 0, 1)

        return torch.from_numpy(noisy)

    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(self.files[idx]), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = torch.from_numpy(img).permute(2,0,1).float() / 255.0
        dark = self.synthesize_dark(img)
        return dark, img  # (synthetic dark, normal light)