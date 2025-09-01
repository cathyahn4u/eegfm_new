# -*- coding: utf-8 -*-
"""
dummy_train_test.py
[오류 수정] main.py와 동일하게, config의 device 정책('cuda_if_available' 등)을
먼저 해석하여 최종 장치 이름('cuda' 또는 'cpu')을 결정한 뒤,
torch.device()에 전달하도록 로직을 수정하여 RuntimeError를 해결합니다.
"""
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np
import importlib

from src.config_schema import (
    AppConfig, SystemConfig, ModelContainerConfig, CBraModConfig,
    PatchEncoderConfig, PositionalEncoderConfig, BackboneConfig,
    ClassificationHeadConfig, ReconstructionHeadConfig,
    PretrainStrategyConfig, MAEConfig, TrainingConfig,
    PretrainTrainingConfig, FinetuneTrainingConfig, DataHandlingConfig
)
from src.tasks.pretrain_task import run_pretrain
from src.tasks.finetune_task import run_finetune

def get_dummy_config() -> AppConfig:
    """테스트용 AppConfig 데이터클래스 객체를 생성합니다."""
    return AppConfig(
        mode='finetune',
        dataset_name='DummyDataset',
        system=SystemConfig(seed=42, device='cuda_if_available'),
        model_selection='CBraMod',
        model=ModelContainerConfig(
            CBraMod=CBraModConfig(
                embedding_dim=32,
                patch_encoder=PatchEncoderConfig(name='TimeFreqPatchEncoder', patch_length=100, out_channels=8),
                positional_encoder=PositionalEncoderConfig(name='AsymmetricConditionalPositionalEncoder', kernel_size_spatial=3, kernel_size_temporal=3),
                backbone=BackboneConfig(name='CBraModBackbone', depth=2, heads=4, mlp_dim=64, dropout=0.1),
                classification_head=ClassificationHeadConfig(name='ClassificationHead', pooling_mode='mean', fc_depth=2, hidden_dim=64, dropout=0.1)
            )
        ),
        pretrain_strategy=PretrainStrategyConfig(
            name='MAE',
            MAE=MAEConfig(
                mask_ratio=0.75,
                reconstruction_head=ReconstructionHeadConfig(name='ReconstructionHead')
            )
        ),
        training=TrainingConfig(
            pretrain=PretrainTrainingConfig(epochs=1, batch_size=4, learning_rate=1e-4, weight_decay=0.05),
            finetune=FinetuneTrainingConfig(epochs=1, batch_size=4, learning_rate=1e-4, weight_decay=0.01)
        ),
        data_handling=DataHandlingConfig(num_workers=4, common_channels=[]),
        datasets={'properties': {'DummyDataset': {'num_classes': 4, 'channels': ['C3', 'C4', 'Cz', 'Fz'], 'sampling_rate': 100, 'duration_secs': 2}}}
    )

def create_dummy_dataloaders(config: AppConfig):
    props = config.datasets['properties']['DummyDataset']
    num_channels = len(props['channels'])
    seq_len = int(props['sampling_rate'] * props['duration_secs'])
    num_classes = props['num_classes']
    num_samples = 40

    X = torch.randn(num_samples, num_channels, seq_len)
    y = torch.randint(0, num_classes, (num_samples,))
    dataset = TensorDataset(X, y)
    
    train_size, val_size = int(0.6 * num_samples), int(0.2 * num_samples)
    test_size = num_samples - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    batch_size = config.training.finetune.batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Dummy data created: {num_samples} samples, {num_channels} channels, {seq_len} time points.")
    print(f"DataLoaders created: Train={len(train_loader.dataset)}, Val={len(val_loader.dataset)}, Test={len(test_loader.dataset)}")
    
    return train_loader, val_loader, test_loader

def instantiate_model(config: AppConfig):
    model_name = config.model_selection
    model_config = getattr(config.model, model_name)
    pretrain_config = config.pretrain_strategy if config.mode == 'pretrain' else None

    dataset_props = config.datasets['properties'][config.dataset_name]
    model_config.num_channels = len(dataset_props['channels'])
    model_config.num_patches_per_channel = int(dataset_props['sampling_rate'] * dataset_props['duration_secs'] / model_config.patch_encoder.patch_length)
    model_config.num_classes = dataset_props['num_classes']
    
    model_module = importlib.import_module(f"src.models.{model_name.lower()}")
    ModelClass = getattr(model_module, model_name)
    
    model = ModelClass(model_config, pretrain_strategy_config=pretrain_config)
    return model

def get_device(device_policy: str) -> torch.device:
    """Config의 장치 정책을 해석하여 torch.device 객체를 반환합니다."""
    if device_policy == 'cuda_if_available':
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_policy == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available, but config is set to 'cuda'.")
        return torch.device("cuda")
    else:
        return torch.device("cpu")

if __name__ == '__main__':
    print("="*50)
    print(" EEG Foundation Model Platform: Dummy Pipeline Test")
    print("="*50)

    dummy_config = get_dummy_config()
    device = get_device(dummy_config.system.device)
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = create_dummy_dataloaders(dummy_config)
    
    print("\n--- Testing Pre-training Pipeline (1 Epoch) ---")
    dummy_config.mode = 'pretrain'
    pretrain_model = instantiate_model(dummy_config).to(device)
    run_pretrain(pretrain_model, train_loader, val_loader, dummy_config.training.pretrain, device)
    print("✅ Pre-training pipeline test completed successfully.")

    print("\n--- Testing Fine-tuning Pipeline (1 Epoch) ---")
    dummy_config.mode = 'finetune'
    finetune_model = instantiate_model(dummy_config).to(device)
    run_finetune(finetune_model, train_loader, val_loader, test_loader, dummy_config.training.finetune, device)
    print("✅ Fine-tuning pipeline test completed successfully.")

