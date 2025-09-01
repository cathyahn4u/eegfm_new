# -*- coding: utf-8 -*-
"""
학습 및 평가 태스크에서 공통으로 사용되는 함수들을 정의하는 모듈.
코드의 중복을 줄이고 재사용성을 높입니다.
"""
import torch
from tqdm import tqdm
import numpy as np
from src.utils.metrics import calculate_metrics

def run_one_epoch(model, data_loader, criterion, optimizer, device, mode, is_train=True):
    """
    한 번의 에폭(epoch) 동안의 학습 또는 검증을 수행합니다.

    Args:
        model (torch.nn.Module): 학습/검증할 모델.
        data_loader (DataLoader): 데이터 로더.
        criterion (torch.nn.Module): 손실 함수.
        optimizer (torch.optim.Optimizer): 옵티마이저 (학습 시에만 사용).
        device (torch.device): 연산을 수행할 장치.
        mode (str): 모델의 forward 모드 ('pretrain' 또는 'finetune').
        is_train (bool): 학습 모드 여부. True이면 가중치를 업데이트합니다.

    Returns:
        float: 해당 에폭의 평균 손실.
    """
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0
    
    desc = "Training" if is_train else "Validating"
    progress_bar = tqdm(data_loader, desc=f"Epoch Step [{desc}]", leave=False)

    # is_train이 False일 경우 그래디언트 계산 비활성화
    with torch.set_grad_enabled(is_train):
        # 사전학습의 경우 라벨이 없음
        data_iter = iter(progress_bar)
        for _ in range(len(data_loader)):
            batch_data, batch_labels = next(data_iter)
            batch_data = batch_data.to(device)
            
            if is_train:
                optimizer.zero_grad()

            # --- Forward Pass ---
            if mode == 'pretrain':
                loss = model(batch_data, mode=mode)
            else: # 'finetune'
                batch_labels = batch_labels.to(device)
                outputs = model(batch_data, mode=mode)
                loss = criterion(outputs, batch_labels)
            
            # --- Backward Pass & Optimization ---
            if is_train:
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(data_loader)


def evaluate(model, data_loader, criterion, device, mode='finetune'):
    """
    주어진 데이터셋에 대해 모델을 평가하고 손실과 성능 지표를 반환합니다.

    Args:
        model (torch.nn.Module): 평가할 모델.
        data_loader (DataLoader): 평가용 데이터 로더.
        criterion (torch.nn.Module): 손실 함수.
        device (torch.device): 연산을 수행할 장치.
        mode (str): 모델의 forward 모드.

    Returns:
        tuple: (평균 손실, 성능 지표 딕셔너리)
    """
    model.eval()
    total_loss = 0
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for batch_data, batch_labels in tqdm(data_loader, desc="Evaluating", leave=False):
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            
            outputs = model(batch_data, mode=mode)
            loss = criterion(outputs, batch_labels)
            total_loss += loss.item()
            
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_labels.extend(batch_labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    metrics = calculate_metrics(np.array(all_labels), np.array(all_preds), probs=np.array(all_probs))
    
    return avg_loss, metrics
