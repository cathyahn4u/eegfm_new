# -*- coding: utf-8 -*-
"""
components.py
[기능 추가] ClassificationHead가 config 파일에 명시된 'activation' 값을
읽어, nn.ReLU, nn.ELU, nn.GELU 등 다양한 활성화 함수를 동적으로
생성하도록 수정합니다.
"""
import torch
import torch.nn as nn
import math
from src.config_schema import (
    PatchEncoderConfig, PositionalEncoderConfig, ClassificationHeadConfig,
    ReconstructionHeadConfig, BackboneConfig
)

class TimeFreqPatchEncoder(nn.Module):
    """EEG 신호를 시간 및 주파수 도메인에서 패치로 만들고 임베딩합니다."""
    def __init__(self, config: PatchEncoderConfig):
        super().__init__()
        self.patch_length = config.patch_length
        self.embedding_dim = config.embedding_dim
        
        self.time_conv = nn.Sequential(
            nn.Conv1d(1, config.out_channels, kernel_size=49, stride=25, padding=24),
            nn.BatchNorm1d(config.out_channels), nn.GELU(),
            nn.Conv1d(config.out_channels, config.out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(config.out_channels), nn.GELU(),
            nn.Conv1d(config.out_channels, config.out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(config.out_channels), nn.GELU(),
        )
        
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, self.patch_length)
            dummy_output = self.time_conv(dummy_input)
            conv_out_size = dummy_output.flatten(1).shape[1]

        self.time_fc = nn.Linear(conv_out_size, self.embedding_dim)
        self.freq_fc = nn.Linear(self.patch_length // 2 + 1, self.embedding_dim)

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        return x.unfold(2, self.patch_length, self.patch_length)

    def embed(self, patches: torch.Tensor) -> torch.Tensor:
        B, C, N, P = patches.shape
        x_flat = patches.view(B * C * N, 1, P)
        
        time_feat = self.time_conv(x_flat).flatten(1)
        time_feat = self.time_fc(time_feat)
        
        fft_feat = torch.fft.rfft(x_flat, dim=-1).abs()
        freq_feat = self.freq_fc(fft_feat.squeeze(1))
        
        combined = (time_feat + freq_feat).view(B, C * N, self.embedding_dim)
        return combined

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_patched = self.patchify(x)
        return self.embed(x_patched)

class AsymmetricConditionalPositionalEncoder(nn.Module):
    def __init__(self, config: PositionalEncoderConfig):
        super().__init__()
        self.num_channels = config.num_channels
        self.num_patches_per_channel = config.num_patches_per_channel
        self.embedding_dim = config.embedding_dim
        
        self.pos_conv = nn.Conv2d(
            self.embedding_dim, self.embedding_dim,
            kernel_size=(config.kernel_size_spatial, config.kernel_size_temporal),
            padding=(config.kernel_size_spatial // 2, config.kernel_size_temporal // 2),
            groups=self.embedding_dim
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        C, N = self.num_channels, self.num_patches_per_channel
        
        if L != C * N:
            raise ValueError(f"Input sequence length {L} does not match C*N ({C*N}).")
        
        x_2d = x.view(B, C, N, D).permute(0, 3, 1, 2)
        pos_embedding = self.pos_conv(x_2d).permute(0, 2, 3, 1).contiguous().view(B, L, D)
        return x + pos_embedding

class CrissCrossAttention(nn.Module):
    def __init__(self, config: BackboneConfig):
        super().__init__()
        self.embedding_dim = config.embedding_dim
        self.num_heads = config.heads
        self.num_channels = config.num_channels
        self.num_patches_per_channel = config.num_patches_per_channel
        
        assert self.embedding_dim % self.num_heads == 0, "Embedding dim must be divisible by heads"
        self.head_dim = self.embedding_dim // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.to_qkv = nn.Linear(self.embedding_dim, self.embedding_dim * 3, bias=False)
        self.to_out = nn.Linear(self.embedding_dim, self.embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        C, N = self.num_channels, self.num_patches_per_channel
        
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(B, L, self.num_heads, self.head_dim).transpose(1, 2), qkv)

        # Spatial Attention
        q_s, k_s, v_s = map(lambda t: t.view(B, self.num_heads, C, N, self.head_dim), (q, k, v))
        q_s, k_s, v_s = q_s.permute(0, 1, 3, 2, 4), k_s.permute(0, 1, 3, 4, 2), v_s.permute(0, 1, 3, 2, 4)
        dots_s = (q_s @ k_s) * self.scale
        attn_s = dots_s.softmax(dim=-1)
        out_s = (attn_s @ v_s).permute(0, 1, 3, 2, 4).reshape(B, self.num_heads, L, self.head_dim)

        # Temporal Attention
        q_t, k_t, v_t = map(lambda t: t.view(B, self.num_heads, C, N, self.head_dim), (q, k, v))
        dots_t = (q_t @ k_t.transpose(-1, -2)) * self.scale
        attn_t = dots_t.softmax(dim=-1)
        out_t = (attn_t @ v_t).reshape(B, self.num_heads, L, self.head_dim)
        
        out = out_s + out_t
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.to_out(out)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class CrissCrossTransformerBlock(nn.Module):
    def __init__(self, config: BackboneConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.embedding_dim)
        self.attn = CrissCrossAttention(config)
        self.norm2 = nn.LayerNorm(config.embedding_dim)
        self.ffn = FeedForward(config.embedding_dim, config.mlp_dim, config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class ClassificationHead(nn.Module):
    """다양한 풀링 전략과 활성화 함수를 지원하는 분류 헤드."""
    def __init__(self, config: ClassificationHeadConfig):
        super().__init__()
        self.pooling_mode = config.pooling_mode
        self.embedding_dim = config.embedding_dim
        
        activation_fn_map = {
            'relu': nn.ReLU(),
            'elu': nn.ELU(),
            'gelu': nn.GELU()
        }
        self.activation = activation_fn_map.get(config.activation.lower())
        if self.activation is None:
            raise ValueError(f"Unsupported activation function: {config.activation}")

        if self.pooling_mode == 'cls_token':
            self.cls_token = nn.Parameter(torch.randn(1, 1, self.embedding_dim))
            input_dim = self.embedding_dim
        elif self.pooling_mode == 'flatten':
            input_dim = self.embedding_dim * config.num_patches
        else:
            input_dim = self.embedding_dim

        layers = []
        if config.fc_depth == 1:
            layers.append(nn.Linear(input_dim, config.num_classes))
        else:
            layers.append(nn.Linear(input_dim, config.hidden_dim))
            layers.append(self.activation)
            layers.append(nn.Dropout(config.dropout))
            layers.append(nn.Linear(config.hidden_dim, config.num_classes))
        
        self.fc = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pooling_mode == 'mean':
            pooled = x.mean(dim=1)
        elif self.pooling_mode == 'flatten':
            pooled = x.flatten(1)
        else:
            pooled = x.mean(dim=1)

        return self.fc(pooled)

class ReconstructionHead(nn.Module):
    def __init__(self, config: ReconstructionHeadConfig):
        super().__init__()
        self.decoder = nn.Linear(config.embedding_dim, config.patch_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)

