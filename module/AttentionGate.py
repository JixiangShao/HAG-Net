import torch
from torch import nn
import torch.nn.functional as F


class AttentionGate(nn.Module):
    """Enhanced Attention Gate"""

    def __init__(self, encoder_channels, decoder_channels):
        super().__init__()

        # Encoder feature compression
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(encoder_channels, decoder_channels, kernel_size=1),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True)
        )

        # Decoder feature transformation
        self.decoder_conv = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels, kernel_size=1),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True)
        )

        # Attention calculation module
        self.attention = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels // 4, encoder_channels, kernel_size=1),
            nn.Sigmoid()
        )

        # Vascular structure enhancement branch
        self.vessel_structure = nn.Sequential(
            nn.Conv2d(encoder_channels, encoder_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(encoder_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(encoder_channels // 2, encoder_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, encoder_feat, decoder_feat):
        # Transform encoder and decoder feature maps
        encoder_proj = self.encoder_conv(encoder_feat)
        decoder_proj = self.decoder_conv(decoder_feat)

        # Fuse transformed encoder and decoder features
        fused = torch.relu(encoder_proj + decoder_proj)

        # Generate base attention weight map
        base_att = self.attention(fused)

        # Generate vessel structural attention weights
        vessel_att = self.vessel_structure(encoder_feat)

        # Multiply two attention maps to get composite attention
        combined_att = base_att * vessel_att

        # Apply composite attention to encoder features
        attended_encoder = encoder_feat * combined_att

        # Concatenate weighted encoder features with decoder features, preserve original channel dimensions
        return torch.cat([attended_encoder, decoder_feat], dim=1)