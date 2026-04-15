import sys
sys.path.append("/content/CV_Project")

import torch
import matplotlib.pyplot as plt
import numpy as np
import random
from pytorch_msssim import ssim

from src.model import UNetEnhancer
from src.data import LowLightDataset


# ── CONFIG ──────────────────────────────────────────
LOW_DIR = "/content/CV_Project/dataset/train/low"
HIGH_DIR = "/content/CV_Project/dataset/train/high"

BASELINE_PATH = "checkpoints/unet_step1_pretrained.pth"
HYBRID_PATH   = "checkpoints/unet_hybrid.pth"

NUM_IMAGES = 3


# ── METRICS ─────────────────────────────────────────
def compute_psnr(output, target):
    mse = torch.mean((output - target) ** 2)
    return 10 * torch.log10(1.0 / (mse + 1e-8))


# ── LOAD MODEL ──────────────────────────────────────
def load_model(path, device):
    model = UNetEnhancer(use_freq_block=True).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


# ── MAIN FUNCTION ───────────────────────────────────
def visualize():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = LowLightDataset(LOW_DIR, HIGH_DIR, img_size=256)

    model_base = load_model(BASELINE_PATH, device)
    model_hybrid = load_model(HYBRID_PATH, device)

    indices = random.sample(range(len(dataset)), NUM_IMAGES)

    for idx in indices:
        low, high = dataset[idx]

        low  = low.unsqueeze(0).to(device)
        high = high.unsqueeze(0).to(device)

        with torch.no_grad():
            out_base = model_base(low)
            out_hybrid = model_hybrid(low)

        # ── METRICS ──
        def get_metrics(out):
            psnr = compute_psnr(out, high).item()
            ssim_val = ssim(out, high, data_range=1.0).item()
            return psnr, ssim_val

        psnr_b, ssim_b = get_metrics(out_base)
        psnr_h, ssim_h = get_metrics(out_hybrid)

        # ── CONVERT TO IMAGE ──
        def to_img(x):
            x = x.detach().squeeze().permute(1, 2, 0).cpu().numpy()
            return np.clip(x, 0, 1)

        imgs = [
            ("Low Light", to_img(low)),
            (f"Baseline\nPSNR:{psnr_b:.2f}\nSSIM:{ssim_b:.3f}", to_img(out_base)),
            (f"Hybrid\nPSNR:{psnr_h:.2f}\nSSIM:{ssim_h:.3f}", to_img(out_hybrid)),
            ("Ground Truth", to_img(high))
        ]

        # ── PLOT ──
        plt.figure(figsize=(12, 4))

        for i, (title, img) in enumerate(imgs):
            plt.subplot(1, 4, i + 1)
            plt.imshow(img)
            plt.title(title, fontsize=10)
            plt.axis("off")

        plt.tight_layout()
        plt.show()


# ── RUN ─────────────────────────────────────────────
if __name__ == "__main__":
    visualize()