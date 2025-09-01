# -*- coding: utf-8 -*-
"""
mumtaz.py
Mumtaz et al. (2016) MDD-NC 데이터셋을 위한 로더 및 전처리기.
"""
import mne
import numpy as np
import os
from ..base_dataset import BaseDataset
from ..preprocessing import resample, create_epochs, normalize, bandpass_filter

class MumtazLoader(BaseDataset):
    def _load_data(self):
        """
        MDD 환자와 정상 대조군(.mat 파일)을 로드합니다.
        """
        data_dict = {}
        labels_dict = {}
        
        # 데이터는 'MDD'와 'NC' 폴더로 나뉨
        mdd_path = os.path.join(self.base_path, 'MDD')
        nc_path = os.path.join(self.base_path, 'NC')
        
        subj_counter = 1
        
        # MDD 환자 데이터 로드 (label=1)
        for fname in sorted(os.listdir(mdd_path)):
            if fname.endswith('.mat'):
                # 이 데이터셋은 .mat 파일이지만, MNE로 읽을 수 있는 구조가 아님
                # 예시를 위해 더미 Raw 객체 생성 로직을 사용
                # 실제로는 scipy.io.loadmat을 사용하여 데이터를 읽고 MNE Raw 객체로 변환해야 함
                n_channels = len(self.props['montage']['channels'])
                n_times = self.props['sampling_rate'] * 180 # 3분 데이터 가정
                sfreq = self.props['sampling_rate']
                data = np.random.randn(n_channels, n_times)
                ch_names = self.props['montage']['channels']
                info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
                raw = mne.io.RawArray(data, info)
                
                data_dict[subj_counter] = raw
                labels_dict[subj_counter] = 1 # MDD
                subj_counter += 1

        # 정상 대조군 데이터 로드 (label=0)
        for fname in sorted(os.listdir(nc_path)):
            if fname.endswith('.mat'):
                n_channels = len(self.props['montage']['channels'])
                n_times = self.props['sampling_rate'] * 180
                sfreq = self.props['sampling_rate']
                data = np.random.randn(n_channels, n_times)
                ch_names = self.props['montage']['channels']
                info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
                raw = mne.io.RawArray(data, info)
                
                data_dict[subj_counter] = raw
                labels_dict[subj_counter] = 0 # NC
                subj_counter += 1

        return data_dict, labels_dict

    def _preprocess_data(self, raw, label):
        """
        필터링, 리샘플링, 에포크 생성을 수행합니다.
        """
        params = self.props['preprocess_params']
        raw_filtered = bandpass_filter(raw, params['l_freq'], params['h_freq'])
        raw_resampled = resample(raw_filtered, params['resample_freq'])
        
        epochs = create_epochs(raw_resampled, duration=self.props['duration_secs'])
        
        data = epochs.get_data()
        data = normalize(data)
        
        # 모든 에포크는 동일한 라벨을 가짐
        epoch_labels = np.full(len(epochs), label)
        
        return data, epoch_labels
