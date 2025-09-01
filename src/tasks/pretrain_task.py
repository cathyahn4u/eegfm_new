# -*- coding: utf-8 -*-
"""
pretrain_task.py
[기능 추가] 사전 학습 시작 전, config에 명시된 'checkpoint_path'로부터
이전 학습 가중치를 불러와 이어서 학습하는 로직을 추가합니다.
"""
import torch
import os
from .base_task import run_one_epoch

def run_pretrain(model, train_loader, val_loader, config, device):
    """사전 학습 파이프라인을 실행합니다."""

    # --- 체크포인트 로딩 로직 ---
    if config.checkpoint_path:
        try:
            print(f"🔄 Resuming pre-training from checkpoint: {config.checkpoint_path}")
            checkpoint = torch.load(config.checkpoint_path, map_location=device)
            
            # 사전 학습은 보통 동일한 구조에서 이어가지만, 유연성을 위해 strict=False 사용
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
            
            print("✅ Checkpoint loaded successfully.")
            if missing_keys:
                print(f"   - Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"   - Unexpected keys: {unexpected_keys}")

        except FileNotFoundError:
            print(f"⚠️ Checkpoint file not found at {config.checkpoint_path}. Starting pre-training from scratch.")
        except Exception as e:
            print(f"⚠️ An error occurred while loading checkpoint: {e}. Starting pre-training from scratch.")
    else:
        print("ℹ️ No checkpoint path provided. Starting pre-training from scratch.")


    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    print(f"Starting pre-training for {config.epochs} epochs...")
    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(config.epochs):
        avg_train_loss = run_one_epoch(model, train_loader, None, optimizer, device, mode='pretrain', is_train=True)
        avg_val_loss = run_one_epoch(model, val_loader, None, None, device, mode='pretrain', is_train=False)
        
        print(f"Epoch {epoch+1}/{config.epochs} | Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            print(f"🚀 New best validation loss found: {best_val_loss:.4f}")

    print("Pre-training finished.")

    # --- 모델 저장 로직 ---
    if best_model_state and config.save_path:
        save_dir = os.path.dirname(config.save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        torch.save(best_model_state, config.save_path)
        print(f"✅ Best pre-trained model saved to {config.save_path}")

