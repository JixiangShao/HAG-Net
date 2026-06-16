import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """LayerNorm adapted for small batches, replacing BatchNorm to avoid statistical bias with limited fundus image data"""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        # x: [B, C, H, W]
        mean = x.mean(dim=1, keepdim=True)  # Calculate mean along channel dimension
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class CMUNeXtBlock(nn.Module):
    """High-frequency branch optimization: improve capability to capture fine vessel edges"""
    def __init__(self, ch_in, ch_out):
        super().__init__()
        # Attention branch: ensure intermediate channels are at least 1 during channel compression
        self.att_part = nn.Sequential(
            # Key modification: use max to guarantee intermediate channel count ≥ 1
            nn.Conv2d(ch_in, max(ch_in // 4, 1), kernel_size=1),  # Channel compression
            nn.Conv2d(max(ch_in // 4, 1), ch_in, kernel_size=1),  # Channel recovery
            nn.Sigmoid()
        )
        # Convolution branch: avoid excessive channel expansion and ensure intermediate channels ≥ 1
        self.conv_part = nn.Sequential(
            nn.Conv2d(ch_in, max(ch_in * 2, 1), kernel_size=3, padding=1, groups=ch_in),  # Depthwise convolution
            LayerNorm(max(ch_in * 2, 1)),  # Adapt to modified channel count
            nn.GELU(),
            nn.Conv2d(max(ch_in * 2, 1), ch_in, kernel_size=1),  # Pointwise convolution
            LayerNorm(ch_in)
        )
        self.up = nn.Conv2d(ch_in, ch_out, kernel_size=1)  # Unify output channels

    def forward(self, x):
        res = x
        att = self.att_part(x)
        x = self.conv_part(x)
        x = x * att + res  # Attention-weighted residual connection
        return self.up(x)


class MEEM(nn.Module):
    """Low-frequency branch optimization: preserve main vessel structures and suppress background noise"""

    def __init__(self, in_dim, hidden_dim, width=2):
        super().__init__()
        self.width = width
        # Input compression: better focus on core low-frequency information
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, kernel_size=3, padding=1),  # 3x3 convolution to enhance local correlation
            LayerNorm(hidden_dim),
            nn.GELU()  # Replace Sigmoid to prevent vanishing gradients
        )

        # Multi-scale pooling: capture low-frequency structures with different receptive fields
        self.pools = nn.ModuleList([
            nn.AvgPool2d(kernel_size=2 ** i, stride=1, padding=2 ** (i - 1))  # Gradually increasing pooling kernel size
            for i in range(1, width)
        ])

        self.mid_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
                LayerNorm(hidden_dim),
                nn.GELU()
            ) for _ in range(width - 1)
        ])

        # Enhanced edge preservation: replace standard convolution with Sobel operator
        self.edge_enhance = nn.Conv2d(
            hidden_dim, hidden_dim,
            kernel_size=3, padding=1,
            groups=hidden_dim,
            bias=False
        )
        # Initialize edge enhancement convolution with Sobel operators (horizontal + vertical)
        sobel = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32)
        sobel = sobel.repeat(hidden_dim, 1, 1, 1)  # Share identical Sobel kernel across all channels
        self.edge_enhance.weight.data = sobel
        self.edge_enhance.weight.requires_grad = False  # Freeze edge extraction operator

        # Output feature fusion: reduce information loss during channel compression
        self.out_conv = nn.Sequential(
            nn.Conv2d(hidden_dim * width, in_dim, kernel_size=3, padding=1),  # 3x3 convolution to restore spatial correlation
            LayerNorm(in_dim),
            nn.GELU()
        )

    def forward(self, x):
        mid = self.in_conv(x)
        out = [mid]

        for i in range(self.width - 1):
            mid = self.pools[i](mid)  # Multi-scale pooling operation
            mid = self.mid_convs[i](mid)
            mid = mid + 0.1 * self.edge_enhance(mid)  # Mild edge enhancement to avoid blurred low-frequency features
            # Align feature size with input to prevent dimension shift caused by pooling
            mid = F.interpolate(mid, size=x.shape[2:], mode='bilinear', align_corners=True)
            out.append(mid)

        return self.out_conv(torch.cat(out, dim=1))


class ChannelAttention(nn.Module):
    """Dynamically balance high-frequency and low-frequency weights to focus on vessel regions"""

    def __init__(self, dim, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        mid_channels = max(dim // reduction, 1)  # Guarantee at least one intermediate channel even if dim < reduction
        self.fc = nn.Sequential(
            nn.Conv2d(dim, mid_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(mid_channels, dim, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(self.avg_pool(x))


class HLAEM(nn.Module):
    """Optimized HLAEM: enhance fine vessel high-frequency details + stabilize low-frequency structures + dynamic weight balancing"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # High-low frequency separation (softer separation strategy to avoid over-filtering fine vessel features)
        self.down = nn.AvgPool2d(kernel_size=2, stride=1, padding=1)  # Adjust pooling kernel to reduce information loss

        # High-frequency branch: strengthen fine vessel edges (deeper CMUNeXtBlock)
        self.high_branch = CMUNeXtBlock(ch_in=dim, ch_out=dim)

        # Low-frequency branch: retain main vessel trunks (reduced MEEM width)
        self.low_branch = MEEM(in_dim=dim, hidden_dim=dim // 2, width=2)

        # Dynamic fusion gate: adaptively balance contributions from high and low frequency branches
        self.gate = nn.Sequential(
            nn.Conv2d(2 * dim, dim, kernel_size=1),
            LayerNorm(dim),
            nn.Sigmoid()
        )

        # Channel attention: highlight feature channels related to blood vessels
        self.att = ChannelAttention(dim)

        # Residual connection + feature recalibration
        self.final_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x):
        # 1. Fine-grained high-low frequency separation to avoid loss of tiny vessels
        low = self.down(x)  # Low-frequency component (main vessel trunks)
        high = x - F.interpolate(low, size=x.shape[2:], mode='bilinear', align_corners=True)  # High-frequency component (edges & fine details)

        # 2. Feature enhancement for two branches
        low_enhanced = self.low_branch(low)  # Low-frequency enhancement (main structural vessels)
        high_enhanced = self.high_branch(high)  # High-frequency enhancement (fine vessel edges)

        # 3. Dynamic fusion: adjust high/low frequency weights based on feature content
        low_upsampled = F.interpolate(low_enhanced, size=x.shape[2:], mode='bilinear', align_corners=True)
        fusion = torch.cat([high_enhanced, low_upsampled], dim=1)
        gate_weight = self.gate(fusion)  # Generate weight map for high/low frequency branches (range: 0~1)
        fused = high_enhanced * gate_weight + low_upsampled * (1 - gate_weight)  # Gated feature fusion

        # 4. Attention enhancement + residual skip connection
        out = self.att(fused)  # Emphasize vessel-relevant feature channels
        out = self.final_conv(out) + x  # Residual path to preserve original input features

        return out


# Test output dimension of modules
if __name__ == "__main__":
    x = torch.randn(1, 64, 256, 256)  # Simulate intermediate feature map from UNet (B=1, C=64, H=256, W=256)
    hlaem = HLAEM(dim=64)
    out = hlaem(x)
    print(f"Input shape: {x.shape}, Output shape: {out.shape}")  # Output dimension should match input