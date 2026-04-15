import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch_msssim import ssim
import torchvision.models as models

from src.model import UNetEnhancer
from src.data import LowLightDataset, SyntheticLowLightDataset

# ── Config ──────────────────────────────────────────
BATCH_SIZE  = 4
EPOCHS      = 40
PRETRAIN_EPOCHS = 10
LR          = 1e-4
IMG_SIZE    = 256

TRAIN_LOW   = "/content/CV_Project/dataset/train/low"
TRAIN_HIGH  = "/content/CV_Project/dataset/train/high"
SYNTHETIC_DIR = "/content/CV_Project/dataset/normal_images"

# ── Hybrid Loss ─────────────────────────────────────
class HybridLoss(nn.Module):
    def __init__(self, ssim_weight=0.5, perceptual_weight=0.1, freq_weight=0.1):
        super().__init__()

        self.ssim_weight = ssim_weight
        self.perceptual_weight = perceptual_weight
        self.freq_weight = freq_weight

        self.l1 = nn.L1Loss(reduction='none')

        # VGG for perceptual loss
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features[:16]
        self.vgg = vgg.eval()
        for p in self.vgg.parameters():
            p.requires_grad = False

    def forward(self, output, target, input_low):
        # ── Noise-adaptive L1 ──
        brightness = input_low.mean(dim=1, keepdim=True)
        weight = (1.0 - brightness + 0.1)
        weight = weight / weight.mean()

        l1_map = self.l1(output, target).mean(dim=1, keepdim=True)
        weighted_l1 = (weight * l1_map).mean()

        # ── SSIM ──
        ssim_loss = 1.0 - ssim(output, target, data_range=1.0, size_average=True)

        # ── Perceptual Loss ──
        feat_out = self.vgg(output)
        feat_target = self.vgg(target)
        perceptual_loss = F.l1_loss(feat_out, feat_target)

        # ── Frequency Loss ──
        fft_out = torch.fft.fft2(output)
        fft_target = torch.fft.fft2(target)
        freq_loss = F.l1_loss(torch.abs(fft_out), torch.abs(fft_target))

        # ── Total Loss ──
        total_loss = (
            weighted_l1 +
            self.ssim_weight * ssim_loss +
            self.perceptual_weight * perceptual_loss +
            self.freq_weight * freq_loss
        )

        return total_loss


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Model ─────────────────────────────
    model = UNetEnhancer(use_freq_block=True).to(device)

    # ── Loss ──────────────────────────────
    criterion = HybridLoss(ssim_weight=0.5, perceptual_weight=0.1, freq_weight=0.1)
    criterion.vgg = criterion.vgg.to(device)  # IMPORTANT

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # =========================================================
    # 🔥 PHASE 1: SYNTHETIC PRETRAINING
    # =========================================================
    print("\n🔥 Phase 1: Synthetic Pretraining")

    synthetic_dataset = SyntheticLowLightDataset(SYNTHETIC_DIR, IMG_SIZE)
    synthetic_loader  = DataLoader(synthetic_dataset, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(PRETRAIN_EPOCHS):
        model.train()
        total_loss = 0

        for low, high in synthetic_loader:
            low, high = low.to(device), high.to(device)

            optimizer.zero_grad()
            out = model(low)
            loss = criterion(out, high, low)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg = total_loss / len(synthetic_loader)
        print(f"[Pretrain] Epoch [{epoch+1}/{PRETRAIN_EPOCHS}] Loss: {avg:.4f}")

    # =========================================================
    # 🚀 PHASE 2: REAL TRAINING
    # =========================================================
    print("\n🚀 Phase 2: Training on Real Dataset")

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

        print(f"[Train] Epoch [{epoch+1}/{EPOCHS}] Loss: {avg:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

    # ── Save Model ────────────────────────
    torch.save(model.state_dict(), "checkpoints/unet_hybrid.pth")
    print("✅ Saved: checkpoints/unet_hybrid.pth")


if __name__ == "__main__":
    train()