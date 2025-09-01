# -*- coding: utf-8 -*-
"""
cbramodbackbone.py
[오류 수정] CBraModBackbone 클래스가 자신의 핵심 파라미터인 embedding_dim을
생성자에서 self.embedding_dim 속성으로 저장하도록 수정합니다.
이를 통해 이 백본에 의존하는 다른 모듈(e.g., MAEStrategy)이 필요한 정보를
안정적으로 조회할 수 있게 하여 AttributeError를 근본적으로 해결합니다.
"""
import torch.nn as nn
from src.models.components import CrissCrossTransformerBlock
from src.config_schema import BackboneConfig

class CBraModBackbone(nn.Module):
    """
    CBraMod의 핵심 Transformer 인코더.
    CrissCrossTransformerBlock을 여러 층으로 쌓아 구성됩니다.
    """
    def __init__(self, config: BackboneConfig):
        """
        CBraModBackbone을 초기화합니다.

        Args:
            config (BackboneConfig): 백본 관련 모든 설정을 담은 데이터클래스.
        """
        super().__init__()
        # --- 근본적인 오류 해결 ---
        # 백본의 핵심 속성인 embedding_dim을 인스턴스 변수로 저장합니다.
        self.embedding_dim = config.embedding_dim
        
        self.layers = nn.ModuleList([
            CrissCrossTransformerBlock(config) for _ in range(config.depth)
        ])
        self.norm = nn.LayerNorm(config.embedding_dim)

    def forward(self, x):
        """
        입력 텐트를 Transformer 블록들을 순차적으로 통과시킵니다.

        Args:
            x (torch.Tensor): (B, L, D) 형태의 입력 텐서.

        Returns:
            torch.Tensor: 인코딩된 (B, L, D) 형태의 텐서.
        """
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

