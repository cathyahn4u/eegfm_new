# -*- coding: utf-8 -*-
"""
config_schema.py
[오류 수정] default.yaml과의 일관성을 위해 PretrainTrainingConfig와
FinetuneTrainingConfig에 'save_path' 필드를 명시적으로 추가합니다.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

# --- System & Data ---
@dataclass
class SystemConfig:
    device: str
    seed: int

@dataclass
class DataHandlingConfig:
    num_workers: int
    common_channels: List[str]
    process: Dict[str, List[str]] = field(default_factory=dict)

# --- Model Components ---
@dataclass
class PatchEncoderConfig:
    name: str
    patch_length: int
    out_channels: int
    embedding_dim: Optional[int] = None

@dataclass
class PositionalEncoderConfig:
    name: str
    kernel_size_spatial: int
    kernel_size_temporal: int
    embedding_dim: Optional[int] = None
    num_channels: Optional[int] = None
    num_patches_per_channel: Optional[int] = None

@dataclass
class BackboneConfig:
    name: str
    depth: int
    heads: int
    mlp_dim: int
    dropout: float
    embedding_dim: Optional[int] = None
    num_channels: Optional[int] = None
    num_patches_per_channel: Optional[int] = None

@dataclass
class ClassificationHeadConfig:
    name: str
    pooling_mode: str
    fc_depth: int
    hidden_dim: int
    dropout: float
    activation: str = 'relu'
    embedding_dim: Optional[int] = None
    num_patches: Optional[int] = None
    num_classes: Optional[int] = None

@dataclass
class ReconstructionHeadConfig:
    name: str
    embedding_dim: Optional[int] = None
    patch_length: Optional[int] = None

# --- Top-level Model Config ---
@dataclass
class CBraModConfig:
    embedding_dim: int
    patch_encoder: PatchEncoderConfig
    positional_encoder: PositionalEncoderConfig
    backbone: BackboneConfig
    classification_head: ClassificationHeadConfig
    num_channels: Optional[int] = None
    num_patches_per_channel: Optional[int] = None
    num_classes: Optional[int] = None

@dataclass
class ModelContainerConfig:
    CBraMod: CBraModConfig

# --- Learning Strategies ---
@dataclass
class MAEConfig:
    mask_ratio: float
    reconstruction_head: ReconstructionHeadConfig
    embedding_dim: Optional[int] = None

@dataclass
class PretrainStrategyConfig:
    name: str
    MAE: MAEConfig

# --- Training Configs ---
@dataclass
class PretrainTrainingConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    save_path: str = './pretrained_model.pth'
    checkpoint_path: Optional[str] = None

@dataclass
class FinetuneTrainingConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    checkpoint_path: Optional[str] = None
    save_path: Optional[str] = './finetuned_model.pth'

@dataclass
class TrainingConfig:
    pretrain: PretrainTrainingConfig
    finetune: FinetuneTrainingConfig

# --- Root Config Schema ---
@dataclass
class AppConfig:
    mode: str
    dataset_name: str
    system: SystemConfig
    model_selection: str
    model: ModelContainerConfig
    pretrain_strategy: PretrainStrategyConfig
    training: TrainingConfig
    data_handling: DataHandlingConfig
    datasets: Dict[str, Any]

