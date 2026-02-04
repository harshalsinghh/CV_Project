import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import UNetEnhancer
from data import LowLightDataset
from pytorch_msssim import ssim

# ------------------
# Config
# ------------------
BATCH_SIZE = 4
EPOCHS = 30
LR = 1e-4
IMG_SIZE = 256
SSIM_WEIGHT = 0.5

TRAIN_LOW = "dataset/train/low"
TRAIN_HIGH = "dataset/train/high"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Torch CUDA available:", torch.cuda.is_available())
print("Using device:", device)

# ------------------
# Dataset & Loader
# ------------------
dataset = LowLightDataset(
    low_dir=TRAIN_LOW,
    high_dir=TRAIN_HIGH,
    img_size=IMG_SIZE
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=False
)

# ------------------
# Model, Loss, Optim
# ------------------
model = UNetEnhancer().to(device)
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ------------------
# Training Loop
# ------------------
model.train()
for epoch in range(EPOCHS):
    epoch_loss = 0.0

    for i, (low, high) in enumerate(loader):
        if epoch == 0 and i == 0:
            print("First batch loaded. Training has started.")

        low = low.to(device)
        high = high.to(device)

        optimizer.zero_grad()
        output = model(low)

        l1_loss = criterion(output, high)
        ssim_loss = 1 - ssim(output, high, data_range=1.0, size_average=True)

        loss = l1_loss + SSIM_WEIGHT * ssim_loss

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(loader)
    print(f"Epoch [{epoch+1}/{EPOCHS}] - Avg Loss: {avg_loss:.4f}")

# ------------------
# Save Model
# ------------------
os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/unet_enhancer.pth")
print("Model saved to checkpoints/unet_enhancer.pth")
