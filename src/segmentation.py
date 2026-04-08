import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision import transforms
from src.model import UNetEnhancer
from src.data import LowLightDataset

def run_segmentation_eval(checkpoint_path, low_dir, high_dir,
                           use_freq_block=True, n_samples=5):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load enhancer
    enhancer = UNetEnhancer(use_freq_block=use_freq_block).to(device)
    enhancer.load_state_dict(torch.load(checkpoint_path, map_location=device))
    enhancer.eval()

    # Load segmentation model (pretrained, no training needed)
    seg_model = deeplabv3_resnet50(pretrained=True).to(device)
    seg_model.eval()

    dataset = LowLightDataset(low_dir, high_dir, img_size=256)

    # ImageNet normalization for DeepLab
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    def segment(img_tensor):
        """img_tensor: [C,H,W] in [0,1]"""
        inp = normalize(img_tensor).unsqueeze(0).to(device)
        with torch.no_grad():
            out = seg_model(inp)['out']
        return out.argmax(1).squeeze().cpu().numpy()

    fig, axes = plt.subplots(n_samples, 4, figsize=(20, n_samples * 5))
    fig.suptitle("Segmentation: Dark vs Enhanced vs Ground Truth",
                 fontsize=16, fontweight='bold', y=1.01)

    titles = ["Low-Light Input", "Enhanced Output",
              "Seg (Dark)", "Seg (Enhanced)"]

    for i in range(n_samples):
        low, high = dataset[i]

        with torch.no_grad():
            enhanced = enhancer(low.unsqueeze(0).to(device)).squeeze(0).cpu()

        seg_dark     = segment(low)
        seg_enhanced = segment(enhanced)

        imgs = [
            low.permute(1,2,0).numpy(),
            enhanced.permute(1,2,0).numpy(),
            seg_dark,
            seg_enhanced,
        ]

        for j, (ax, img, title) in enumerate(zip(axes[i], imgs, titles)):
            if j < 2:
                ax.imshow(np.clip(img, 0, 1))
            else:
                ax.imshow(img, cmap='tab20')
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.axis('off')

    plt.tight_layout()
    plt.savefig("segmentation_comparison.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: segmentation_comparison.png")