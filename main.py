# -*- coding: utf-8 -*-
"""
main.py
[아키텍처 최종 수정] 프로젝트의 메인 실행 로직을 전체 워크플로우를
관리하는 컨트롤러로 재설계합니다. 이제 이 파일은 config의 'mode'에 따라
'process' 목록에 정의된 모든 데이터셋에 대해 해당 작업을 순차적으로
반복 실행합니다. 또한, 경로의 플레이스홀더를 동적으로 해석합니다.
"""
import argparse
import torch
import numpy as np
import random
import importlib
from src.utils.helpers import load_config, resolve_paths
from src.data_handling.data_loader import get_dataloaders
from src.config_schema import AppConfig, CBraModConfig

def set_seed(seed: int):
    """모든 난수 생성기의 시드를 고정하여 재현성을 확보합니다."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

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

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="EEG Foundation Model Training Platform")
    parser.add_argument('--config', type=str, required=True, help="Path to the experiment config file")
    args = parser.parse_args()

    # --- 1. 기본 설정 로드 ---
    config: AppConfig = load_config(args.config)
    set_seed(config.system.seed)
    device = get_device(config.system.device)
    print(f"Global settings loaded. Using device: {device}")
    print(f"Running in mode: '{config.mode}'")

    # --- 2. 모드에 따라 실행할 데이터셋 목록 결정 ---
    if config.mode not in config.data_handling.process:
        raise ValueError(f"Process list for mode '{config.mode}' not found in config.data_handling.process")
    datasets_to_run = config.data_handling.process[config.mode]
    
    # --- 3. 각 데이터셋에 대해 작업 루프 실행 ---
    for dataset_name in datasets_to_run:
        print("\n" + "="*80)
        print(f"Processing dataset: {dataset_name}")
        print("="*80)

        # --- 3.1. 현재 데이터셋에 맞게 설정 객체 복사 및 수정 ---
        # 원본 config를 수정하지 않기 위해 깊은 복사를 사용하는 것이 안전하지만,
        # 여기서는 편의를 위해 직접 수정 후 진행합니다.
        run_config = config
        run_config.dataset_name = dataset_name
        
        # 동적 경로 해석
        run_config = resolve_paths(run_config)

        # --- 3.2. 데이터로더 및 모델 생성 ---
        train_loader, val_loader, test_loader = get_dataloaders(run_config)
        
        model_name = run_config.model_selection
        model_config: CBraModConfig = getattr(run_config.model, model_name)
        
        dataset_props = run_config.datasets['properties'][dataset_name]
        model_config.num_channels = len(dataset_props['montage']['channels'])
        model_config.num_patches_per_channel = int(dataset_props['sampling_rate'] * dataset_props['duration_secs'] / model_config.patch_encoder.patch_length)
        model_config.num_classes = dataset_props['num_classes']
        
        ModelClass = getattr(importlib.import_module(f"src.models.{model_name.lower()}"), model_name)
        
        pretrain_strategy_config = run_config.pretrain_strategy if config.mode in ['pretrain', 'finetune'] else None

        model = ModelClass(model_config, pretrain_strategy_config=pretrain_strategy_config).to(device)
        print(f"Successfully instantiated model '{model_name}' for dataset '{dataset_name}'")

        # --- 3.3. 해당 모드의 태스크 실행 ---
        task_module_name = f"src.tasks.{config.mode}_task"
        run_task_function_name = f"run_{config.mode}"
        
        try:
            task_module = importlib.import_module(task_module_name)
            run_task = getattr(task_module, run_task_function_name)
            
            if config.mode == 'pretrain':
                run_task(model, train_loader, val_loader, run_config.training.pretrain, device)
            elif config.mode == 'finetune':
                run_task(model, train_loader, val_loader, test_loader, run_config.training.finetune, device)
            elif config.mode == 'evaluation':
                 run_task(model, test_loader, run_config.training.finetune, device) # finetune config 재활용
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Could not find or run task for mode '{config.mode}'. Error: {e}")

    print("\n" + "="*80)
    print("All tasks finished successfully.")
    print("="*80)

if __name__ == '__main__':
    main()

