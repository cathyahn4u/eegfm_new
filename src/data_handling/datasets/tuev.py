# -*- coding: utf-8 -*-
"""
tuev.py
TUH EEG Event Corpus (TUEV)를 위한 로더 및 전처리기.
"""
import mne
import numpy as np
import os
from ..base_dataset import BaseDataset
from ..preprocessing import resample, create_epochs, normalize, bandpass_filter

class TUEVLoader(BaseDataset):
    def _load_data(self):
        """
        TUEV 데이터셋을 로드합니다.
        이 데이터셋은 복잡한 주석 구조를 가집니다.
        여기서는 단순화된 로더를 구현합니다.
        """
        # TUAB와 유사하게, 제공된 train/test split을 사용
        # 피험자 단위 분할 대신 파일 기반으로 처리
        
        data_dict = {}
        # 라벨은 raw 객체의 annotation에서 추출
        labels_dict = {i: None for i in range(10000)} # 임시 크기
        
        file_counter = 0
        for root, _, files in os.walk(self.base_path):
            for f in files:
                if f.endswith('.edf'):
                    fpath = os.path.join(root, f)
                    try:
                        raw = mne.io.read_raw_edf(fpath, preload=True, verbose='ERROR')
                        data_dict[file_counter] = raw
                        file_counter += 1
                    except Exception as e:
                         print(f"Warning: Could not read file {fpath}. Error: {e}")
        
        return data_dict, labels_dict

    def _preprocess_data(self, raw, labels):
        """
        필터링, 몽타주, 리샘플링, 에포크화 및 라벨링을 수행합니다.
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
        
        # Annotation에서 이벤트 추출
        events, event_id = mne.events_from_annotations(raw_resampled)
        
        # config에 라벨 맵이 있다면 사용, 없다면 자동 생성된 ID 사용
        label_mapping = self.props.get('label_mapping', event_id)
        
        epochs = mne.Epochs(raw_resampled, events, event_id=label_mapping, 
                            tmin=0, tmax=self.props['duration_secs'], 
                            preload=True, baseline=None, on_missing='warn')

        epoch_labels = epochs.events[:, -1]
        
        data = epochs.get_data()
        data = normalize(data)
        
        return data, epoch_labels
