import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pytorch_msssim import ssim
from src.model import UNetEnhancer
from src.data import LowLightDataset

# ── Config ──────────────────────────────────────────
BATCH_SIZE  = 4
EPOCHS      = 40
LR          = 1e-4
IMG_SIZE    = 256
SSIM_WEIGHT = 0.5
TRAIN_LOW   = "dataset/train/low"
TRAIN_HIGH  = "dataset/train/high"

# ── Noise-Adaptive Weighted Loss ────────────────────
class NoiseAdaptiveLoss(nn.Module):
    """
    Spatially weights L1 loss by local darkness.
    Darker pixels get higher weight since they are
    harder to reconstruct and more noise-affected.
    """
    def __init__(self, ssim_weight=0.5):
        super().__init__()
        self.ssim_weight = ssim_weight
        self.l1 = nn.L1Loss(reduction='none')

    def forward(self, output, target, input_low):
        # Brightness map from low-light input
        brightness = input_low.mean(dim=1, keepdim=True)  # [B,1,H,W]

        # Darker = higher weight (floor at 0.1 to avoid instability)
        weight = (1.0 - brightness + 0.1)
        weight = weight / weight.mean()  # normalize so average weight = 1

        # Weighted L1
        l1_map = self.l1(output, target).mean(dim=1, keepdim=True)
        weighted_l1 = (weight * l1_map).mean()

        # SSIM
        ssim_loss = 1.0 - ssim(output, target, data_range=1.0, size_average=True)

        return weighted_l1 + self.ssim_weight * ssim_loss


def train(use_freq_block=True, epochs=EPOCHS, save_name="unet_enhancer.pth"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | FreqBlock: {use_freq_block}")

    dataset = LowLightDataset(TRAIN_LOW, TRAIN_HIGH, IMG_SIZE)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                         num_workers=0, pin_memory=False)

    model     = UNetEnhancer(use_freq_block=use_freq_block).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = NoiseAdaptiveLoss(ssim_weight=SSIM_WEIGHT)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for low, high in loader:
            low, high = low.to(device), high.to(device)
            optimizer.zero_grad()
            out  = model(low)
            loss = criterion(out, high, low)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        avg = total_loss / len(loader)
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg:.4f} LR: {scheduler.get_last_lr()[0]:.6f}")

    torch.save(model.state_dict(), f"checkpoints/{save_name}")
    print(f"Saved: checkpoints/{save_name}")
    return model


if __name__ == "__main__":
    train()