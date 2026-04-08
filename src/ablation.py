import torch
import numpy as np
import matplotlib.pyplot as plt
from src.model import UNetEnhancer
from src.data import LowLightDataset
from pytorch_msssim import ssim

def compute_psnr(output, target):
    mse = torch.mean((output - target) ** 2)
    return 10 * torch.log10(1.0 / (mse + 1e-8))

def ablation_study(low_dir, high_dir):
    """
    Compare 3 variants:
    A: L1 only, no freq block    (train separately with train.py)
    B: L1+SSIM, no freq block    (your original model)
    C: L1+SSIM + freq block      (new model — full proposal)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = LowLightDataset(low_dir, high_dir, img_size=256)

    variants = [
        ("A: L1 Only",          "checkpoints/unet_l1only.pth",    False),
        ("B: L1+SSIM",          "checkpoints/unet_enhancer.pth",  False),
        ("C: L1+SSIM+FreqBlock","checkpoints/unet_freqblock.pth", True),
    ]

    results = {}

    for label, ckpt, use_freq in variants:
        try:
            model = UNetEnhancer(use_freq_block=use_freq).to(device)
            model.load_state_dict(torch.load(ckpt, map_location=device))
            model.eval()

            psnr_list, ssim_list = [], []
            with torch.no_grad():
                for low, high in dataset:
                    low  = low.unsqueeze(0).to(device)
                    high = high.unsqueeze(0).to(device)
                    out  = model(low)
                    psnr_list.append(compute_psnr(out, high).item())
                    ssim_list.append(ssim(out, high, data_range=1.0).item())

            results[label] = {
                'psnr': np.mean(psnr_list),
                'ssim': np.mean(ssim_list)
            }
            print(f"{label} | PSNR: {results[label]['psnr']:.2f} | SSIM: {results[label]['ssim']:.4f}")

        except FileNotFoundError:
            print(f"Checkpoint not found for {label} — skipping")

    # Plot
    if results:
        labels = list(results.keys())
        psnrs  = [results[l]['psnr'] for l in labels]
        ssims  = [results[l]['ssim'] for l in labels]
        colors = ['#e07b39', '#2196F3', '#4CAF50']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("Ablation Study: Component Contribution",
                     fontsize=15, fontweight='bold')

        bars1 = ax1.bar(labels, psnrs, color=colors, width=0.5, edgecolor='black')
        ax1.set_ylabel("PSNR (dB)", fontsize=12)
        ax1.set_title("PSNR Comparison", fontsize=13)
        ax1.set_ylim(min(psnrs) - 1, max(psnrs) + 1)
        for bar, val in zip(bars1, psnrs):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                     f'{val:.2f}', ha='center', fontweight='bold')

        bars2 = ax2.bar(labels, ssims, color=colors, width=0.5, edgecolor='black')
        ax2.set_ylabel("SSIM", fontsize=12)
        ax2.set_title("SSIM Comparison", fontsize=13)
        ax2.set_ylim(min(ssims) - 0.05, max(ssims) + 0.05)
        for bar, val in zip(bars2, ssims):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                     f'{val:.4f}', ha='center', fontweight='bold')

        plt.tight_layout()
        plt.savefig("ablation_study.png", dpi=150, bbox_inches='tight')
        plt.show()
        print("Saved: ablation_study.png")

    return results
