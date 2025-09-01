# -*- coding: utf-8 -*-
"""
tuab.py
TUH Abnormal EEG Corpus (TUAB)를 위한 로더 및 전처리기.
"""
import mne
import numpy as np
import os
from ..base_dataset import BaseDataset
from ..preprocessing import resample, create_epochs, normalize, bandpass_filter

class TUABLoader(BaseDataset):
    def _load_data(self):
        """
        TUAB 데이터셋을 로드합니다. train/eval 폴더 구조를 사용합니다.
        """
        data_dict = {}
        labels_dict = {}
        
        # 이 데이터셋은 train/eval 폴더와 그 아래 normal/abnormal 폴더로 구성됨
        # 피험자 단위 분할 대신, 제공된 분할을 사용
        # 여기서는 단순화를 위해 모든 파일을 하나의 그룹으로 로드
        
        all_files = []
        for root, _, files in os.walk(self.base_path):
            for f in files:
                if f.endswith('.edf'):
                    all_files.append((os.path.join(root, f), 'abnormal' in root))

        # 여기서는 피험자 ID 대신 파일 인덱스를 사용
        for i, (fpath, is_abnormal) in enumerate(all_files):
            try:
                raw = mne.io.read_raw_edf(fpath, preload=True, verbose='ERROR')
                data_dict[i] = raw
                labels_dict[i] = 1 if is_abnormal else 0
            except Exception as e:
                print(f"Warning: Could not read file {fpath}. Error: {e}")

        return data_dict, labels_dict

    def _preprocess_data(self, raw, label):
        """
        필터링, 몽타주, 리샘플링, 에포크화를 수행합니다.
        """
        params = self.props['preprocess_params']
        
        try:
            raw_montaged = self._apply_montage(raw)
        except Exception as e:
            print(f"Warning: Could not apply montage. {e}")
            return np.array([]), np.array([])
        
        raw_filtered = bandpass_filter(raw_montaged, params['l_freq'], params['h_freq'])
        if 'notch_freq' in params:
            raw_filtered.notch_filter(freqs=params['notch_freq'])

        raw_resampled = resample(raw_filtered, params['resample_freq'])
        
        epochs = create_epochs(raw_resampled, duration=self.props['duration_secs'])
        
        data = epochs.get_data()
        data = normalize(data)
        
        epoch_labels = np.full(len(epochs), label)
        
        return data, epoch_labels
