# -*- coding: utf-8 -*-
"""
metrics.py
[오류 수정] numpy 라이브러리가 import되지 않아 발생한 NameError를
해결하기 위해 파일 상단에 'import numpy as np' 구문을 추가합니다.
"""
from sklearn.metrics import cohen_kappa_score, accuracy_score, f1_score, roc_auc_score, average_precision_score
import numpy as np

def calculate_metrics(labels, preds, probs=None):
    """
    주어진 예측과 실제 라벨을 바탕으로 다양한 성능 지표를 계산합니다.

    Args:
        labels (np.array): 실제 라벨.
        preds (np.array): 모델의 예측 라벨.
        probs (np.array, optional): 모델의 예측 확률. Defaults to None.

    Returns:
        dict: 계산된 성능 지표들을 담은 딕셔너리.
    """
    metrics = {}
    
    # 라벨에 존재하는 고유 클래스 수 확인
    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)
    
    metrics['accuracy'] = accuracy_score(labels, preds)
    metrics['cohen_kappa'] = cohen_kappa_score(labels, preds)
    
    if num_classes > 2:
        metrics['weighted_f1'] = f1_score(labels, preds, average='weighted')
    
    # 확률 기반 메트릭 계산
    if probs is not None:
        if num_classes == 2:
            # 이진 분류
            metrics['auroc'] = roc_auc_score(labels, probs[:, 1])
            metrics['auprc'] = average_precision_score(labels, probs[:, 1])
        elif num_classes > 2:
            # 다중 클래스 분류
            try:
                # AUROC (OVO)는 모든 클래스가 라벨에 존재할 때만 계산 가능
                if len(np.unique(labels)) == probs.shape[1]:
                    metrics['auroc_ovo'] = roc_auc_score(labels, probs, multi_class='ovo', average='weighted')
                else:
                    metrics['auroc_ovo'] = 0.0 # 일부 클래스가 배치에 없는 경우
            except ValueError:
                metrics['auroc_ovo'] = 0.0 # 라벨이 한 종류만 있는 경우 등 예외 처리

    return metrics

