import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Two consecutive Conv2d → BatchNorm → ReLU layers."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class EncoderBranch(nn.Module):
    """Three ConvBlock + MaxPool stages.

    Input:  (B, in_channels, 32, 32)
    Output: (B, 128, 4, 4)
    """

    def __init__(self, in_channels):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, 32)
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.pool(self.enc1(x))   # (B, 32, 16, 16)
        x = self.pool(self.enc2(x))   # (B, 64, 8, 8)
        x = self.pool(self.enc3(x))   # (B, 128, 4, 4)
        return x


class DualBranchUNet(nn.Module):
    """Dual-branch UNet for precipitation downsampling.

    Two independent encoder branches process different input modalities:
      - Branch 1 (bands 0-1): IMERG precipitation + GOES BCM
      - Branch 2 (bands 2-4): DEM elevation, slope, aspect

    Features are concatenated at the bottleneck (4×4 spatial), then a
    shared decoder upsamples back to 32×32 producing a 1-channel output.

    Args:
        branch1_in (int): Number of input channels for branch 1 (default 2).
        branch2_in (int): Number of input channels for branch 2 (default 3).
    """

    def __init__(self, branch1_in=2, branch2_in=3):
        super().__init__()

        # Encoders
        self.branch1 = EncoderBranch(branch1_in)
        self.branch2 = EncoderBranch(branch2_in)

        # Bottleneck: 128 + 128 = 256 channels in
        self.bottleneck = ConvBlock(256, 256)

        # Decoder
        self.up1    = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec1   = ConvBlock(128, 128)

        self.up2    = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2   = ConvBlock(64, 64)

        self.up3    = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec3   = ConvBlock(32, 32)

        self.head   = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        # Split input into two modalities
        x1 = x[:, :2, :, :]   # (B, 2, 32, 32)  precip + GOES
        x2 = x[:, 2:, :, :]   # (B, 3, 32, 32)  topo

        # Encode independently
        f1 = self.branch1(x1)  # (B, 128, 4, 4)
        f2 = self.branch2(x2)  # (B, 128, 4, 4)

        # Fuse at bottleneck
        x = torch.cat([f1, f2], dim=1)  # (B, 256, 4, 4)
        x = self.bottleneck(x)           # (B, 256, 4, 4)

        # Decode
        x = self.dec1(self.up1(x))       # (B, 128, 8, 8)
        x = self.dec2(self.up2(x))       # (B, 64, 16, 16)
        x = self.dec3(self.up3(x))       # (B, 32, 32, 32)

        return self.head(x)              # (B, 1, 32, 32)


class DualBranchLoss(nn.Module):
    """Combined masked MSE + total variation loss.

    Args:
        masked_mse_weight (float): Weight for the station-masked MSE term (λ₁).
        tv_weight (float): Weight for the total variation regularization term (λ₂).
    """

    def __init__(self, masked_mse_weight=1.0, tv_weight=0.1):
        super().__init__()
        self.masked_mse_weight = masked_mse_weight
        self.tv_weight = tv_weight

    def forward(self, pred, target, mask):
        """
        Args:
            pred   (Tensor): Model output, shape (B, 1, H, W).
            target (Tensor): Ground-truth station precipitation, shape (B, 1, H, W).
            mask   (Tensor): Boolean station mask, shape (B, 1, H, W).
                             True where ground-truth measurements exist.
        Returns:
            Tensor: Scalar loss value.
        """
        # Masked MSE — only at station pixels
        masked_pred   = pred[mask]
        masked_target = target[mask]
        if masked_pred.numel() > 0:
            mse_loss = torch.mean((masked_pred - masked_target) ** 2)
        else:
            mse_loss = pred.new_tensor(0.0)

        # Total variation — over full prediction to encourage spatial smoothness
        tv_h = torch.mean(torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :]))
        tv_w = torch.mean(torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1]))
        tv_loss = tv_h + tv_w

        total = self.masked_mse_weight * mse_loss + self.tv_weight * tv_loss
        return total, mse_loss, tv_loss
