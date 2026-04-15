import torch
import numpy as np
from pytorch_msssim import ssim
from src.model import UNetEnhancer
from src.data import LowLightDataset

def compute_psnr(output, target):
    mse = torch.mean((output - target) ** 2)
    return 10 * torch.log10(1.0 / (mse + 1e-8))

def evaluate(checkpoint_path, low_dir, high_dir, use_freq_block=True, label="Model"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = UNetEnhancer(use_freq_block=use_freq_block).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    dataset = LowLightDataset(low_dir, high_dir, img_size=256)
    psnr_list, ssim_list = [], []

    with torch.no_grad():
        for low, high in dataset:
            low  = low.unsqueeze(0).to(device)
            high = high.unsqueeze(0).to(device)
            out  = model(low)

            psnr_list.append(compute_psnr(out, high).item())
            ssim_list.append(ssim(out, high, data_range=1.0).item())

    print(f"\n{'='*40}")
    print(f"  {label}")
    print(f"  PSNR: {np.mean(psnr_list):.2f} dB")
    print(f"  SSIM: {np.mean(ssim_list):.4f}")
    print(f"  Images evaluated: {len(dataset)}")
    print(f"{'='*40}")

    return np.mean(psnr_list), np.mean(ssim_list)
if __name__ == "__main__":
    evaluate(
        checkpoint_path="checkpoints/unet_step1_pretrained.pth",
        low_dir="/content/CV_Project/dataset/train/low",
        high_dir="/content/CV_Project/dataset/train/high",
        label="Step1 Model"
    )
