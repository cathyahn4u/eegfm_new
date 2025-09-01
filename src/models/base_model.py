# -*- coding: utf-8 -*-
"""
base_model.py
[아키텍처 최종 수정] 반복되는 설정 오류를 근본적으로 해결하기 위해,
'조립자(Assembler)' 패턴을 완성합니다. 이 모델은 생성 시점에 모든 하위
컴포넌트(Patch Encoder, Backbone, SSL Strategy 등)의 설정(Dataclass)에
모델의 전역 파라미터(embedding_dim 등)를 명시적으로 주입합니다.
이를 통해 정보 흐름을 중앙에서 관리하고, 모든 설정 관련 오류를
원천적으로 차단합니다.
"""
import torch.nn as nn
import importlib
from src.config_schema import CBraModConfig, PretrainStrategyConfig
from dataclasses import is_dataclass

class EEGFoundationBaseModel(nn.Module):
    """
    모든 EEG Foundation Model의 기반이 되는 추상 클래스.
    Config를 기반으로 하위 컴포넌트들을 동적으로 조립하는 '조립자' 역할을 수행합니다.
    """
    def __init__(self, model_config: CBraModConfig, pretrain_strategy_config: PretrainStrategyConfig = None):
        super().__init__()
        self.model_config = model_config
        self.pretrain_strategy_config = pretrain_strategy_config
        
        self.global_params = {
            'embedding_dim': self.model_config.embedding_dim,
            'num_channels': self.model_config.num_channels,
            'num_patches_per_channel': self.model_config.num_patches_per_channel,
            'num_classes': self.model_config.num_classes,
            'patch_length': self.model_config.patch_encoder.patch_length,
            'num_patches': (self.model_config.num_channels or 0) * (self.model_config.num_patches_per_channel or 0)
        }

        components_module = importlib.import_module("src.models.components")
        backbone_module = importlib.import_module(f"src.models.backbones.{self.model_config.backbone.name.lower()}")

        self.patch_encoder = self._create_component(self.model_config.patch_encoder, components_module)
        self.positional_encoder = self._create_component(self.model_config.positional_encoder, components_module)
        self.backbone = self._create_component(self.model_config.backbone, backbone_module)
        self.classification_head = self._create_component(self.model_config.classification_head, components_module)
        
        self.ssl_strategy = None
        if self.pretrain_strategy_config:
            strategy_name = self.pretrain_strategy_config.name
            strategy_config = getattr(self.pretrain_strategy_config, strategy_name)
            self._enrich_config_recursive(strategy_config)
            
            StrategyClass = getattr(importlib.import_module(f"src.learning_strategies.{strategy_name.lower()}"), f"{strategy_name}Strategy")
            
            self.ssl_strategy = StrategyClass(
                config=strategy_config,
                backbone=self.backbone
            )

    def _enrich_config_recursive(self, config_obj):
        """
        데이터클래스 객체와 그 내부에 중첩된 모든 데이터클래스 객체에
        전역 파라미터를 재귀적으로 주입합니다.
        """
        for param, value in self.global_params.items():
            if hasattr(config_obj, param):
                setattr(config_obj, param, value)

        for field_name in config_obj.__dataclass_fields__:
            field_value = getattr(config_obj, field_name)
            if is_dataclass(field_value):
                self._enrich_config_recursive(field_value)

    def _create_component(self, comp_config, module):
        """전역 파라미터가 주입된 설정 객체로 컴포넌트를 생성합니다."""
        self._enrich_config_recursive(comp_config)
        ComponentClass = getattr(module, comp_config.name)
        return ComponentClass(comp_config)

    def _forward_features_and_patches(self, x):
        raw_patches_unflat = self.patch_encoder.patchify(x)
        features_1d = self.patch_encoder.embed(raw_patches_unflat)
        features_1d_pos = self.positional_encoder(features_1d)
        return features_1d_pos, raw_patches_unflat

    def _forward_pretrain(self, x):
        if not self.ssl_strategy:
            raise RuntimeError("Pre-training not configured.")
        
        features, raw_patches = self._forward_features_and_patches(x)
        
        B, C, N, P = raw_patches.shape
        raw_patches_target = raw_patches.view(B, C * N, P)
        
        loss = self.ssl_strategy(features, raw_patches_target)
        return loss

    def _forward_finetune(self, x):
        features, _ = self._forward_features_and_patches(x)
        encoded_features = self.backbone(features)
        logits = self.classification_head(encoded_features)
        return logits

    def forward(self, x, mode='finetune'):
        if mode == 'pretrain':
            return self._forward_pretrain(x)
        elif mode == 'finetune':
            return self._forward_finetune(x)
        else:
            raise ValueError(f"Unknown mode: {mode}")

