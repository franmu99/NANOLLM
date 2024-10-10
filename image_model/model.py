import os
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import math

class ImageProcessor:
    @staticmethod
    def image_to_tensor(image_path):
        img = Image.open(image_path)
        img = img.convert('RGB')
        img_array = np.array(img)
        tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()
        tensor /= 255.0
        return tensor

class Swish(torch.nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class ResNetBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        self.swish = Swish()
        self.shortcut = torch.nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.bn1(x)
        out = self.swish(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.swish(out)
        out = self.conv2(out)
        out += self.shortcut(x)
        return out

class SelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(SelfAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"
        
        self.W_q = nn.Conv2d(d_model, d_model, 1)
        self.W_k = nn.Conv2d(d_model, d_model, 1)
        self.W_v = nn.Conv2d(d_model, d_model, 1)
        self.W_o = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        batch_size, C, H, W = x.shape
        
        Q = self.W_q(x).view(batch_size, self.num_heads, self.head_dim, H * W)
        K = self.W_k(x).view(batch_size, self.num_heads, self.head_dim, H * W)
        V = self.W_v(x).view(batch_size, self.num_heads, self.head_dim, H * W)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        
        context = torch.matmul(attention_weights, V)
        context = context.view(batch_size, C, H, W)
        
        output = self.W_o(context)
        return output, attention_weights

class Downsampling(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super(Downsampling, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.swish = Swish()

    def forward(self, x):
        return self.swish(self.bn(self.conv(x)))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, noise_level=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attention = SelfAttention(d_model, num_heads)
        self.resnet_block = ResNetBlock(d_model, d_model)
        self.downsampling = Downsampling(d_model, d_model * 2)
        self.norm1 = nn.BatchNorm2d(d_model)
        self.norm2 = nn.BatchNorm2d(d_model)
        self.noise_level = noise_level

    def add_noise(self, x):
        noise = torch.randn_like(x) * self.noise_level
        return x + noise

    def forward(self, x, t):
        # Self-attention
        x, attention_weights = self.self_attention(x)
        x = self.norm1(x)
        x = self.add_noise(x)

        # ResNet block
        x = self.resnet_block(x)
        x = self.norm2(x)
        x = self.add_noise(x)

        # Downsampling
        x = self.downsampling(x)
        x = self.add_noise(x)

        return x, attention_weights

class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, num_layers=3, noise_schedule=None):
        super(Encoder, self).__init__()
        d_model = (d_model // num_heads) * num_heads
        self.initial_conv = nn.Conv2d(3, d_model, kernel_size=1)
        
        if noise_schedule is None or len(noise_schedule) < num_layers:
            noise_schedule = [0.1] * num_layers
        
        self.layers = nn.ModuleList([
            EncoderLayer(d_model * (2**i), num_heads, noise_level=noise_schedule[i])
            for i in range(num_layers)
        ])

    def forward(self, x):
        x = self.initial_conv(x)
        attention_weights = []
        for layer in self.layers:
            x, att = layer(x, None)  # Pasamos None como t por ahora
            attention_weights.append(att)
        return x, attention_weights

class DDPMDenoiser(nn.Module):
    def __init__(self, channels, time_emb_dim=32):
        super(DDPMDenoiser, self).__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        
        self.conv1 = nn.Conv2d(channels, channels*2, 3, padding=1)
        self.conv2 = nn.Conv2d(channels*2, channels*2, 3, padding=1)
        self.conv3 = nn.Conv2d(channels*2, channels, 3, padding=1)
        
        self.time_emb1 = nn.Linear(time_emb_dim, channels*2)
        self.time_emb2 = nn.Linear(time_emb_dim, channels*2)

    def forward(self, x, t):
        t = t.unsqueeze(-1).float()
        t = self.time_mlp(t)
        
        h = self.conv1(x)
        h += self.time_emb1(t).unsqueeze(-1).unsqueeze(-1)
        h = F.silu(h)
        
        h = self.conv2(h)
        h += self.time_emb2(t).unsqueeze(-1).unsqueeze(-1)
        h = F.silu(h)
        
        return self.conv3(h)

class DenoisingModule(nn.Module):
    def __init__(self, channels, method='ddpm', noise_level=0.1):
        super(DenoisingModule, self).__init__()
        self.method = method
        self.noise_level = noise_level
        
        if method == 'ddpm':
            self.ddpm_denoiser = DDPMDenoiser(channels)
        elif method == 'wavelet':
            self.wavelet_denoiser = WaveletDenoiser(channels)
        # ... (otros métodos de denoising)

    def forward(self, x, t=None):
        if self.method == 'ddpm':
            return self.ddpm_denoiser(x, t)
        elif self.method == 'wavelet':
            return self.wavelet_denoiser(x)
        # ... (otros métodos de denoising)
        else:
            return x - torch.randn_like(x) * self.noise_level  # Método simple por defecto

class WaveletDenoiser(nn.Module):
    def __init__(self, channels):
        super(WaveletDenoiser, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        # Simulación simple de denoising basado en wavelets
        high_freq = self.conv1(x)
        low_freq = x - high_freq
        denoised_high_freq = torch.sigmoid(self.conv2(high_freq))
        return low_freq + denoised_high_freq

class NonLocalMeansDenoiser:
    def __call__(self, x, h):
        # Implementación simplificada de Non-Local Means
        pad = nn.ReflectionPad2d(1)
        x_pad = pad(x)
        kernel = torch.exp(-torch.arange(3)**2 / (2*h**2))
        kernel = kernel.view(1, 1, -1).repeat(x.size(1), 1, 1)
        return F.conv2d(x_pad, kernel.unsqueeze(-1), groups=x.size(1))

class LearnedDenoiser(nn.Module):
    def __init__(self, channels):
        super(LearnedDenoiser, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels*2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels*2, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = self.conv2(F.relu(self.conv1(x)))
        return x - residual

class DecoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads, noise_level=0.1, denoising_method='ddpm'):
        super(DecoderLayer, self).__init__()
        self.out_channels = (out_channels // num_heads) * num_heads
        self.upsample = nn.ConvTranspose2d(in_channels, self.out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.resnet_block = ResNetBlock(self.out_channels, self.out_channels)
        self.self_attention = SelfAttention(self.out_channels, num_heads)
        self.norm1 = nn.BatchNorm2d(self.out_channels)
        self.norm2 = nn.BatchNorm2d(self.out_channels)
        self.norm3 = nn.BatchNorm2d(self.out_channels)
        self.denoiser = DenoisingModule(self.out_channels, method=denoising_method, noise_level=noise_level)

    def forward(self, x, t):
        # Upsampling
        x = self.upsample(x)
        x = self.norm1(x)
        x = self.denoiser(x, t)

        # ResNet block
        x = self.resnet_block(x)
        x = self.norm2(x)
        x = self.denoiser(x, t)

        # Self-attention
        x, attention_weights = self.self_attention(x)
        x = self.norm3(x)
        x = self.denoiser(x, t)

        return x, attention_weights

class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, num_layers=3, noise_schedule=None, denoising_method='ddpm'):
        super(Decoder, self).__init__()
        d_model = (d_model // num_heads) * num_heads
        
        if noise_schedule is None:
            noise_schedule = [0.1] * num_layers
        
        self.layers = nn.ModuleList([
            DecoderLayer(
                int(d_model * (2**(num_layers-i))), 
                int(d_model * (2**(num_layers-i-1))), 
                num_heads,
                noise_level=noise_schedule[i],
                denoising_method=denoising_method
            )
            for i in range(num_layers)
        ])
        self.final_conv = nn.Conv2d(d_model, 3, kernel_size=1)

    def forward(self, x, t):
        attention_weights = []
        for i, layer in enumerate(self.layers):
            x, att = layer(x, t[i])
            attention_weights.append(att)
        x = self.final_conv(x)
        return x, attention_weights

class ImageAttentionModel(nn.Module):
    def __init__(self, d_model=12, num_heads=4, num_layers=3, noise_schedule=None, denoising_method='ddpm'):
        super(ImageAttentionModel, self).__init__()
        self.d_model = (d_model // num_heads) * num_heads
        self.num_heads = num_heads
        self.num_layers = num_layers
        if noise_schedule is None or len(noise_schedule) < num_layers:
            noise_schedule = [0.1] * num_layers
        self.encoder = Encoder(self.d_model, num_heads, num_layers, noise_schedule)
        self.decoder = Decoder(self.d_model, num_heads, num_layers, noise_schedule[::-1], denoising_method)

    def forward(self, x):
        encoded, encoder_attention = self.encoder(x)
        t = torch.randint(0, 1000, (self.num_layers,)).unsqueeze(1).to(x.device)
        decoded, decoder_attention = self.decoder(encoded, t)
        return decoded, (encoder_attention, decoder_attention)

# Elimina o comenta las siguientes líneas si están presentes al final del archivo
# if __name__ == "__main__":
#     image_path = r'C:\Users\franm\OneDrive\Imágenes\perro_jpeg.jpg'
#     noise_schedule = [0.1, 0.2, 0.3]
#     model = ImageAttentionModel(d_model=12, num_heads=4, num_layers=3, 
#                                 noise_schedule=noise_schedule, denoising_method='ddpm')
#     model.process()