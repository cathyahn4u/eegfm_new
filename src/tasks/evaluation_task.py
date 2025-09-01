# -*- coding: utf-8 -*-
"""
evaluation_task.py
[기능 추가] 미세 조정이 완료된 모델을 불러와 평가만 수행하는
'evaluation' 모드를 위한 새로운 태스크 파일입니다.
"""
import torch
from .base_task import evaluate

def run_evaluation(model, test_loader, config, device):
    """
    저장된 모델 가중치를 불러와 테스트 데이터셋에 대한 평가를 수행합니다.
    
    Args:
        model (torch.nn.Module): 모델 아키텍처.
        test_loader (DataLoader): 테스트 데이터로더.
        config (FinetuneTrainingConfig): 체크포인트 경로가 포함된 설정.
        device (torch.device): 실행 장치.
    """
    # --- 저장된 가중치 로딩 ---
    if config.checkpoint_path:
        try:
            print(f"🔄 Loading fine-tuned weights for evaluation from: {config.checkpoint_path}")
            checkpoint = torch.load(config.checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint)
            print("✅ Weights loaded successfully.")
        except FileNotFoundError:
            print(f"⚠️ ERROR: Checkpoint file not found at {config.checkpoint_path}. Cannot perform evaluation.")
            return
        except Exception as e:
            print(f"⚠️ ERROR: An error occurred while loading weights: {e}. Cannot perform evaluation.")
            return
    else:
        print("⚠️ ERROR: No checkpoint path provided for evaluation mode.")
        return

    # --- 평가 실행 ---
    print("\nStarting evaluation on the test set...")
    criterion = torch.nn.CrossEntropyLoss() # 손실 계산을 위해 필요
    _, test_metrics = evaluate(model, test_loader, criterion, device, mode='finetune')
    
    print("\n--- Final Evaluation Results ---")
    for key, value in test_metrics.items():
        print(f"{key}: {value:.4f}")
    print("--------------------------------")
