import torch
import torch.nn as nn
import torch.nn.functional as F


#Channel Attention Block
class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Ensure the reduced channel dimension is at least 1
        reduced_channel = max(1, channel // reduction)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channel, reduced_channel, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(reduced_channel, channel, 1, bias=False)
        )

    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        return torch.sigmoid(avg_out + max_out)
#Spatial Attention Block
class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttentionModule, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return torch.sigmoid(self.conv(x_cat))
    


#Dual Attention Block
class DualAttentionBlock(nn.Module):
    def __init__(self, channel, reduction=16, kernel_size=7):
        super(DualAttentionBlock, self).__init__()
        self.cam = ChannelAttentionModule(channel, reduction)
        self.sam = SpatialAttentionModule(kernel_size)

    def forward(self, x):
        cam_weights = self.cam(x)
        x = x * cam_weights  # Apply channel attention

        sam_weights = self.sam(x)
        x = x * sam_weights  # Apply spatial attention
        return x
    

#Encoder Block
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.attention = DualAttentionBlock(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.attention(x)
        p = self.pool(x) # output for next level
        return x, p # x is for skip connection, p is for next encoder level
    
#Decoder Block
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.attention = DualAttentionBlock(out_channels)

    def forward(self, x, skip_connection):
        x = self.upsample(x)
        # If the sizes don't match exactly due to padding/stride, crop the skip connection
        diffY = skip_connection.size()[2] - x.size()[2]
        diffX = skip_connection.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x, skip_connection], dim=1)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.attention(x)
        return x
    
#Dual Attention UNet
class DualAttentionUNet(nn.Module):
    def __init__(self, in_channels=5, out_channels=1, features=[8, 16, 32]):
        super(DualAttentionUNet, self).__init__()
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()

        # Encoder
        for feature in features:
            self.encoders.append(EncoderBlock(in_channels, feature))
            in_channels = feature

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features[-1], features[-1] * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features[-1] * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[-1] * 2, features[-1] * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features[-1] * 2),
            nn.ReLU(inplace=True),
            DualAttentionBlock(features[-1] * 2)
        )

        # Decoder (features list is reversed for decoder)
        for i in range(len(features) - 1, -1, -1):
            self.decoders.append(DecoderBlock(features[i] * 2, features[i]))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for encoder in self.encoders:
            skip, x = encoder(x)
            skip_connections.append(skip)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1] # Reverse for decoding

        for idx, decoder in enumerate(self.decoders):
            x = decoder(x, skip_connections[idx])

        return self.final_conv(x)