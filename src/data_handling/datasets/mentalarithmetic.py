# -*- coding: utf-8 -*-
"""
mentalarithmetic.py
Mental Arithmetic 데이터셋 (스트레스 탐지)을 위한 로더 및 전처리기.
"""
import mne
import numpy as np
import os
from ..base_dataset import BaseDataset
from ..preprocessing import resample, create_epochs, normalize, bandpass_filter

class MentalArithmeticLoader(BaseDataset):
    def _load_data(self):
        """
        각 피험자의 .edf 파일을 로드합니다.
        """
        data_dict = {}
        # 이 데이터셋은 라벨이 파일 내 annotation에 포함되어 있음
        labels_dict = {i: None for i in range(1, 37)}
        
        for subj_id in range(1, 37):
            fname = f'Subject{subj_id}.edf'
            fpath = os.path.join(self.base_path, fname)
            if os.path.exists(fpath):
                try:
                    raw = mne.io.read_raw_edf(fpath, preload=True, verbose='ERROR')
                    data_dict[subj_id] = raw
                except Exception as e:
                    print(f"Warning: Could not read file for subject {subj_id}. Error: {e}")
        
        return data_dict, labels_dict

    def _preprocess_data(self, raw, labels):
        """
        필터링, 리샘플링, 에포크 생성 및 라벨링을 수행합니다.
        """
        params = self.props['preprocess_params']
        
        # ECG 채널 제거
        if 'ECG' in raw.ch_names:
            raw.drop_channels(['ECG'])
            
        raw_montaged = self._apply_montage(raw)
        
        raw_filtered = bandpass_filter(raw_montaged, params['l_freq'], params['h_freq'])
        raw_resampled = resample(raw_filtered, params['resample_freq'])
        
        # Annotation에서 이벤트 추출 (1: Rest, 2: Stress)
        events, event_id = mne.events_from_annotations(raw_resampled)
        
        # 0 (no-stress) 과 1 (stress) 로 라벨 매핑
        new_event_id = {'T1': 0, 'T2': 1}
        
        epochs = mne.Epochs(raw_resampled, events, event_id=new_event_id, 
                            tmin=0, tmax=self.props['duration_secs'], 
                            preload=True, baseline=None, on_missing='warn')

        epoch_labels = epochs.events[:, -1]
        
        data = epochs.get_data()
        data = normalize(data)
        
        return data, epoch_labels
