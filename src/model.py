# src/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath

# --- Layer Normalization for 2D Data ---
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.layer_norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x):
        # Permute dimensions to apply LayerNorm on channel dimension
        x = x.permute(0, 2, 3, 1)
        x = self.layer_norm(x)
        # Permute back to original dimension order
        x = x.permute(0, 3, 1, 2)
        return x

# --- CBAM Modules ---
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        hidden_planes = max(in_planes // ratio, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_MLP = nn.Sequential(
            nn.Conv2d(in_planes, hidden_planes, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden_planes, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_MLP(self.avg_pool(x))
        max_out = self.shared_MLP(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=9):
        super(SpatialAttention, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x_cat)
        return self.sigmoid(out)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=8, kernel_size=9):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_planes, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x_out = self.channel_attention(x) * x
        x_out = self.spatial_attention(x_out) * x_out
        return x_out

# --- ConvNeXt Block ---
class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super(ConvNeXtBlock, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.activation = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True) \
            if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)
        x = self.pwconv1(x)
        x = self.activation(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)
        if self.gamma is not None:
            x = self.gamma[:, None, None] * x
        x = self.drop_path(x) + residual
        return x

def create_classification_head(input_dim, num_classes):
    return nn.Sequential(
        nn.Linear(input_dim, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )

# --- Metadata Encoder ---
class MetadataEncoder(nn.Module):
    def __init__(self, input_size, output_size):
        super(MetadataEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * input_size, output_size),
            nn.ReLU()
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # add channel dimension
        x = self.encoder(x)
        return x

# --- Full Model ---
class ConvNextCBAMClassifier(nn.Module):
    def __init__(self, num_classes=6, in_chans=3, metadata_input_size=21, metadata_output_size=768):
        super(ConvNextCBAMClassifier, self).__init__()
        layers = [3, 3, 9, 3]
        dims = [96, 192, 384, 768]
        drop_path_rate = 0.
        layer_scale_init_value = 1e-6

        # Downsampling stem and additional downsampling layers
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0])
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            layer = nn.Sequential(
                LayerNorm2d(dims[i]),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(layer)

        # Stages with ConvNeXt blocks
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(layers))]
        cur = 0
        for i in range(4):
            blocks = []
            for j in range(layers[i]):
                block = ConvNeXtBlock(
                    dim=dims[i],
                    drop_path=dp_rates[cur + j],
                    layer_scale_init_value=layer_scale_init_value
                )
                blocks.append(block)
            self.stages.append(nn.Sequential(*blocks))
            cur += layers[i]

        # CBAM modules after each stage
        self.cbam_modules = nn.ModuleList([CBAM(dim, ratio=8, kernel_size=9) for dim in dims])
        self.norm = nn.LayerNorm(dims[-1])
        self.metadata_encoder = MetadataEncoder(metadata_input_size, dims[-1])
        self.head = create_classification_head(dims[-1] * 2, num_classes)

    def forward_features(self, x):
        for i in range(len(self.downsample_layers)):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            x = self.cbam_modules[i](x)
        return x

    def forward(self, x, metadata):
        x = self.forward_features(x)
        x = x.mean(dim=[2, 3])  # Global average pooling
        x = self.norm(x)
        metadata_features = self.metadata_encoder(metadata)
        fused_features = torch.cat([x, metadata_features], dim=1)
        x = self.head(fused_features)
        return x
