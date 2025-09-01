# -*- coding: utf-8 -*-
"""
physionetmi.py
PhysioNet Motor Imagery/Movement (EEGBCI) 데이터셋을 위한 로더 및 전처리기.
"""
import mne
import numpy as np
import os
from ..base_dataset import BaseDataset
from ..preprocessing import resample, create_epochs, normalize

class PhysioNetMILoader(BaseDataset):
    def _load_data(self):
        """
        MNE의 내장 다운로더를 사용하여 PhysioNet Motor Imagery 데이터셋을 로드합니다.
        """
        data_dict = {}
        labels_dict = {}
        
        # 109명의 피험자
        subject_ids = range(1, 110)
        # Motor imagery: fists, feet
        runs = [3, 4, 7, 8, 11, 12] 
        
        for subj_id in subject_ids:
            try:
                # MNE가 데이터를 자동으로 다운로드하고 경로를 관리합니다.
                fnames = mne.datasets.eegbci.load_data(subj_id, runs, path=self.base_path, verbose='ERROR')
                raws = [mne.io.read_raw_edf(f, preload=True, verbose='ERROR') for f in fnames]
                raw_combined = mne.concatenate_raws(raws)
                
                # EOG 채널 제거
                eog_channels = [ch for ch in raw_combined.ch_names if 'EOG' in ch]
                if eog_channels:
                    raw_combined.drop_channels(eog_channels)

                data_dict[subj_id] = raw_combined
                
                # 라벨은 이벤트에서 추출되므로 여기서는 None으로 둡니다.
                labels_dict[subj_id] = None

            except Exception as e:
                print(f"Warning: Could not process files for subject {subj_id}. Error: {e}")
        
        return data_dict, labels_dict

    def _preprocess_data(self, raw, labels):
        """
        리샘플링, 에포크 생성 및 정규화를 수행합니다.
        """
        raw_montaged = self._apply_montage(raw)
        raw_resampled = resample(raw_montaged, self.props['preprocess_params']['resample_freq'])
        
        events, event_id = mne.events_from_annotations(raw_resampled)
        # T0: rest, T1: left fist, T2: right fist
        # 논문에서는 4-class (left, right, both fists, both feet)를 언급했으나,
        # 이 데이터셋의 표준 실행은 보통 2-class 또는 3-class. 여기서는 이벤트 ID 기반으로 처리.
        picks = mne.pick_types(raw_resampled.info, meg=False, eeg=True, stim=False)
        
        # 이벤트 ID 기반으로 에포크 생성
        epochs = mne.Epochs(raw_resampled, events, event_id, tmin=0, tmax=self.props['duration_secs'], 
                            proj=True, picks=picks, baseline=None, preload=True)
        
        epoch_labels = epochs.events[:, -1]
        
        data = epochs.get_data()
        data = normalize(data)
        
        return data, epoch_labels
