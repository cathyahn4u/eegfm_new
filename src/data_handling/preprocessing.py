# -*- coding: utf-8 -*-
"""
preprocessing.py
[오류 수정] 모든 데이터 로더가 공통으로 사용하는 전처리 유틸리티 함수인
'bandpass_filter', 'resample', 'create_epochs', 'normalize'를
이 파일에 모두 정의하여 ImportError를 근본적으로 해결합니다.
"""
import mne
import numpy as np

def bandpass_filter(raw: mne.io.Raw, l_freq: float, h_freq: float) -> mne.io.Raw:
    """MNE Raw 객체에 밴드패스 필터를 적용합니다."""
    return raw.copy().filter(l_freq=l_freq, h_freq=h_freq, fir_design='firwin', skip_by_annotation='edge', verbose=False)

def resample(raw: mne.io.Raw, new_freq: int) -> mne.io.Raw:
    """MNE Raw 객체를 새로운 샘플링 주파수로 리샘플링합니다."""
    return raw.copy().resample(sfreq=new_freq, verbose=False)

def create_epochs(raw: mne.io.Raw, duration: float, overlap: float = 0) -> mne.Epochs:
    """MNE Raw 객체를 고정된 길이의 에포크로 분할합니다."""
    return mne.make_fixed_length_epochs(raw, duration=duration, overlap=overlap, preload=True, verbose=False)

def normalize(data: np.ndarray) -> np.ndarray:
    """
    데이터를 채널별로 Z-score 정규화합니다.
    입력 데이터 형태: (n_epochs, n_channels, n_times)
    """
    mean = np.mean(data, axis=-1, keepdims=True)
    std = np.std(data, axis=-1, keepdims=True)
    # 0으로 나누는 것을 방지하기 위해 작은 epsilon 값을 더합니다.
    return (data - mean) / (std + 1e-8)

