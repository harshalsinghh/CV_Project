import torch
import torch.nn as nn
import torch.fft

class FrequencyAwareBlock(nn.Module):
    """
    Splits features into low/high frequency components,
    processes them separately, then recombines.
    Inserted at the bottleneck of U-Net.
    """
    def __init__(self, channels):
        super().__init__()
        self.low_branch = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.high_branch = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # FFT — work in float32 for stability
        x_float = x.float()
        fft = torch.fft.fft2(x_float, norm='ortho')
        magnitude = torch.abs(fft)
        phase = torch.angle(fft)

        # Low frequency mask (center of spectrum)
        h, w = x.shape[-2:]
        low_mask = torch.zeros_like(magnitude)
        low_mask[:, :, h//4:3*h//4, w//4:3*w//4] = 1.0
        high_mask = 1.0 - low_mask

        # Process each frequency band
        low_feat  = self.low_branch(magnitude * low_mask)
        high_feat = self.high_branch(magnitude * high_mask)

        # Fuse and reconstruct
        fused = self.fuse(torch.cat([low_feat, high_feat], dim=1))
        new_fft = torch.polar(fused, phase)
        out = torch.real(torch.fft.ifft2(new_fft, norm='ortho'))

        return out.to(x.dtype)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)


class UNetEnhancer(nn.Module):
    def __init__(self, use_freq_block=True):
        super().__init__()
        self.use_freq_block = use_freq_block

        # Encoder
        self.enc1 = ConvBlock(3, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(256, 512)

        # Frequency-Aware Block at bottleneck
        if use_freq_block:
            self.freq_block = FrequencyAwareBlock(512)

        # Decoder
        self.up3   = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3  = ConvBlock(512, 256)
        self.up2   = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2  = ConvBlock(256, 128)
        self.up1   = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1  = ConvBlock(128, 64)

        self.output = nn.Sequential(
            nn.Conv2d(64, 3, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Encode
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        # Bottleneck
        b = self.bottleneck(self.pool(e3))

        # Frequency block
        if self.use_freq_block:
            b = self.freq_block(b)

        # Decode with skip connections
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.output(d1)