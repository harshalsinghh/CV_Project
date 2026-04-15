import sys
sys.path.append("/content/CV_Project")  # FIX: ensures imports work in Colab

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch_msssim import ssim
import torchvision.models as models

from src.model import UNetEnhancer
from src.data import LowLightDataset

# ── Config ──────────────────────────────────────────
BATCH_SIZE  = 4
EPOCHS      = 10   # fast ablation
LR          = 1e-4
IMG_SIZE    = 256

TRAIN_LOW   = "/content/CV_Project/dataset/train/low"
TRAIN_HIGH  = "/content/CV_Project/dataset/train/high"

# 🔥 CHANGE THIS ONLY
MODE = "perceptual"   # baseline / perceptual / frequency / hybrid

# ── Hybrid Loss ─────────────────────────────────────
class HybridLoss(nn.Module):
    def __init__(self, ssim_weight=0.5, perceptual_weight=0.0, freq_weight=0.0):
        super().__init__()

        self.ssim_weight = ssim_weight
        self.perceptual_weight = perceptual_weight
        self.freq_weight = freq_weight

        self.l1 = nn.L1Loss(reduction='none')

        # Only load VGG if needed (saves time)
        if perceptual_weight > 0:
            vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features[:16]
            self.vgg = vgg.eval()
            for p in self.vgg.parameters():
                p.requires_grad = False
        else:
            self.vgg = None

    def forward(self, output, target, input_low):
        # ── Noise-adaptive L1 ──
        brightness = input_low.mean(dim=1, keepdim=True)
        weight = (1.0 - brightness + 0.1)
        weight = weight / weight.mean()

        l1_map = self.l1(output, target).mean(dim=1, keepdim=True)
        weighted_l1 = (weight * l1_map).mean()

        # ── SSIM ──
        ssim_loss = 1.0 - ssim(output, target, data_range=1.0, size_average=True)

        total_loss = weighted_l1 + self.ssim_weight * ssim_loss

        # ── Perceptual ──
        if self.perceptual_weight > 0:
            feat_out = self.vgg(output)
            feat_target = self.vgg(target)
            perceptual_loss = F.l1_loss(feat_out, feat_target)
            total_loss += self.perceptual_weight * perceptual_loss

        # ── Frequency ──
        if self.freq_weight > 0:
            fft_out = torch.fft.fft2(output)
            fft_target = torch.fft.fft2(target)
            freq_loss = F.l1_loss(torch.abs(fft_out), torch.abs(fft_target))
            total_loss += self.freq_weight * freq_loss

        return total_loss


# ── Loss Selector ───────────────────────────────────
def get_loss(mode):
    if mode == "baseline":
        return HybridLoss(0.5, 0.0, 0.0), "unet_baseline.pth"
    elif mode == "perceptual":
        return HybridLoss(0.5, 0.1, 0.0), "unet_perceptual.pth"
    elif mode == "frequency":
        return HybridLoss(0.5, 0.0, 0.1), "unet_frequency.pth"
    elif mode == "hybrid":
        return HybridLoss(0.5, 0.1, 0.1), "unet_hybrid.pth"
    else:
        raise ValueError("Invalid MODE")


# ── Training ────────────────────────────────────────
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Mode: {MODE}")

    model = UNetEnhancer(use_freq_block=True).to(device)

    criterion, save_name = get_loss(MODE)

    # Move VGG to GPU if used
    if criterion.vgg is not None:
        criterion.vgg = criterion.vgg.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    dataset = LowLightDataset(TRAIN_LOW, TRAIN_HIGH, IMG_SIZE)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for low, high in loader:
            low, high = low.to(device), high.to(device)

            optimizer.zero_grad()
            out = model(low)
            loss = criterion(out, high, low)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        avg = total_loss / len(loader)

        print(f"[{MODE}] Epoch [{epoch+1}/{EPOCHS}] Loss: {avg:.4f}")

    torch.save(model.state_dict(), f"checkpoints/{save_name}")
    print(f"✅ Saved: checkpoints/{save_name}")


if __name__ == "__main__":
    train()