import torch
import numpy as np
import matplotlib.pyplot as plt
from src.model import UNetEnhancer
from src.data import LowLightDataset

class GradCAMLite:
    """
    Lightweight Grad-CAM for image-to-image networks.
    Targets the bottleneck layer.
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None

        # Hook the bottleneck
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.model.bottleneck.register_forward_hook(forward_hook)
        self.model.bottleneck.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor):
        self.model.zero_grad()
        output = self.model(input_tensor)

        # Use mean output as scalar target
        target = output.mean()
        target.backward()

        # Weight activations by gradients
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)

        # Resize to input size
        cam = torch.nn.functional.interpolate(
            cam, size=input_tensor.shape[-2:], mode='bilinear', align_corners=False
        )
        cam = cam.squeeze().cpu().numpy()

        # Normalize
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def run_gradcam(checkpoint_path, low_dir, high_dir,
                use_freq_block=True, n_samples=4):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = UNetEnhancer(use_freq_block=use_freq_block).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    gradcam = GradCAMLite(model)
    dataset = LowLightDataset(low_dir, high_dir, img_size=256)

    fig, axes = plt.subplots(n_samples, 3, figsize=(15, n_samples * 5))
    fig.suptitle("Grad-CAM: Where Does the Model Focus?",
                 fontsize=16, fontweight='bold')

    for i in range(n_samples):
        low, high = dataset[i]
        inp = low.unsqueeze(0).to(device).requires_grad_(True)

        # Generate CAM
        cam = gradcam.generate(inp)

        # Enhanced image
        with torch.no_grad():
            enhanced = model(inp).squeeze(0).cpu().permute(1,2,0).numpy()

        low_np = low.permute(1,2,0).numpy()

        # Overlay heatmap
        heatmap = plt.cm.jet(cam)[:, :, :3]
        overlay = 0.5 * low_np + 0.5 * heatmap
        overlay = np.clip(overlay, 0, 1)

        for ax, img, title in zip(
            axes[i],
            [low_np, overlay, enhanced],
            ["Low-Light Input", "Grad-CAM Attention", "Enhanced Output"]
        ):
            ax.imshow(np.clip(img, 0, 1))
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.axis('off')

    plt.tight_layout()
    plt.savefig("gradcam_analysis.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: gradcam_analysis.png")