"""
CNN Autoencoder for 28x28 grayscale alphanumeric character images (EMNIST).

Encoder: 28x28 -> 14x14 -> 7x7 -> 1x1 (feature map) -> latent vector
Decoder: latent vector -> 1x1 -> 7x7 -> 14x14 -> 28x28
"""

import torch
import torch.nn as nn


class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim

        # ---- Encoder ----
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),   # 28 -> 14
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 14 -> 7
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=7),                       # 7 -> 1
            nn.ReLU(inplace=True),
        )
        self.flatten = nn.Flatten()
        self.fc_enc = nn.Linear(64, latent_dim)

        # ---- Decoder ----
        self.fc_dec = nn.Linear(latent_dim, 64)
        self.unflatten = nn.Unflatten(1, (64, 1, 1))
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=7),                                   # 1 -> 7
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # 7 -> 14
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),   # 14 -> 28
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder_conv(x)
        x = self.flatten(x)
        return self.fc_enc(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc_dec(z)
        x = self.unflatten(x)
        return self.decoder_conv(x)

    def forward(self, x: torch.Tensor):
        latent = self.encode(x)
        recon = self.decode(latent)
        return recon, latent
