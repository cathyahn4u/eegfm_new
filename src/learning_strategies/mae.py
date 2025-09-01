# -*- coding: utf-8 -*-
"""
mae.py
[아키텍처 최종 수정] MAEStrategy가 자신의 내부 구현에 필요한
ReconstructionHead를 직접 생성하고 관리하도록 변경합니다.
이를 통해 사전 학습 전략의 캡슐화를 완성하고, base_model의 역할을
더욱 명확하게 분리합니다.
"""
import torch
import torch.nn as nn
import importlib
from src.config_schema import MAEConfig

class MAEStrategy(nn.Module):
    """
    Masked Autoencoder (MAE) 자가 학습 전략을 구현합니다.
    """
    def __init__(self, config: MAEConfig, backbone: nn.Module):
        """
        MAEStrategy를 초기화합니다.

        Args:
            config (MAEConfig): MAE 관련 모든 설정을 담은 데이터클래스.
            backbone (nn.Module): 특징을 인코딩할 Transformer 백본.
        """
        super().__init__()
        self.config = config
        self.mask_ratio = config.mask_ratio
        self.backbone = backbone

        # --- 근본적인 해결책: ReconstructionHead를 여기서 직접 생성 ---
        components_module = importlib.import_module("src.models.components")
        ReconstructionHeadClass = getattr(components_module, config.reconstruction_head.name)
        self.reconstruction_head = ReconstructionHeadClass(config.reconstruction_head)
        
        # backbone으로부터 직접 embedding_dim을 가져와 mask_token을 생성
        self.mask_token = nn.Parameter(torch.randn(1, 1, self.backbone.embedding_dim))
        self.criterion = nn.MSELoss()

    def forward(self, embedded_features: torch.Tensor, raw_patches_target: torch.Tensor) -> torch.Tensor:
        B, L, D = embedded_features.shape
        P = raw_patches_target.shape[-1]
        
        len_mask = int(L * self.mask_ratio)
        noise = torch.rand(B, L, device=embedded_features.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_mask = ids_shuffle[:, :len_mask]

        masked_features = embedded_features.clone()
        mask_token_expanded = self.mask_token.expand(B, len_mask, D)
        masked_features.scatter_(dim=1, index=ids_mask.unsqueeze(-1).expand(-1, -1, D), src=mask_token_expanded)

        encoded_features = self.backbone(masked_features)

        encoded_masked_features = torch.gather(encoded_features, dim=1, index=ids_mask.unsqueeze(-1).expand(-1, -1, D))
        reconstructed_patches = self.reconstruction_head(encoded_masked_features)
        
        target_masked_patches = torch.gather(raw_patches_target, dim=1, index=ids_mask.unsqueeze(-1).expand(-1, -1, P))

        loss = self.criterion(reconstructed_patches, target_masked_patches)
        return loss

