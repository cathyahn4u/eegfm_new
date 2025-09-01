# -*- coding: utf-8 -*-
"""
finetune_task.py
[기능 추가] 미세 조정한 모델의 가중치를 저장하는 로직을 추가합니다.
"""
import torch
import torch.optim as optim
import os
from .base_task import run_one_epoch, evaluate

def run_finetune(model, train_loader, val_loader, test_loader, config, device):
    """미세 조정 파이프라인을 실행합니다."""

    # --- 사전 학습된 가중치 로딩 로직 ---
    if config.checkpoint_path:
        try:
            print(f"🔄 Loading pre-trained weights from: {config.checkpoint_path}")
            checkpoint = torch.load(config.checkpoint_path, map_location=device)
            
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
            
            print("✅ Weights loaded successfully.")
            if missing_keys:
                print(f"   - Missing keys (expected in fine-tune model): {missing_keys}")
            if unexpected_keys:
                print(f"   - Unexpected keys (in checkpoint but not model): {unexpected_keys}")

        except FileNotFoundError:
            print(f"⚠️ Checkpoint file not found at {config.checkpoint_path}. Training from scratch.")
        except Exception as e:
            print(f"⚠️ An error occurred while loading weights: {e}. Training from scratch.")
            
    else:
        print("ℹ️ No checkpoint path provided. Training from scratch.")

    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_metric = -1
    best_model_state = None

    print(f"Starting fine-tuning for {config.epochs} epochs...")

    for epoch in range(config.epochs):
        train_loss = run_one_epoch(model, train_loader, criterion, optimizer, device, mode='finetune', is_train=True)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device, mode='finetune')

        current_val_metric = val_metrics.get('cohen_kappa', val_metrics.get('auroc', 0))

        print(f"Epoch {epoch+1}/{config.epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Kappa/AUROC: {current_val_metric:.4f}")

        if current_val_metric > best_val_metric:
            best_val_metric = current_val_metric
            best_model_state = model.state_dict()
            print(f"🚀 New best model found at epoch {epoch+1} with metric: {best_val_metric:.4f}")

    if best_model_state:
        # --- 모델 저장 로직 ---
        if config.save_path:
            save_dir = os.path.dirname(config.save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)
            torch.save(best_model_state, config.save_path)
            print(f"\n✅ Best fine-tuned model saved to {config.save_path}")

        print("\nLoading best model and evaluating on the test set...")
        model.load_state_dict(best_model_state)
        _, test_metrics = evaluate(model, test_loader, criterion, device, mode='finetune')
        
        print("\n--- Final Test Results ---")
        for key, value in test_metrics.items():
            print(f"{key}: {value:.4f}")
        print("--------------------------")
    else:
        print("\nNo best model was saved. Skipping final test evaluation.")

