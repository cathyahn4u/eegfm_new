# -*- coding: utf-8 -*-
"""
isruc.py
ISRUC Sleep Staging 데이터셋을 위한 로더 및 전처리기.
수면 단계(5-class) 분류를 위한 시퀀스 데이터를 생성합니다.
"""
import mne
import numpy as np
import os
from ..base_dataset import BaseDataset
from ..preprocessing import bandpass_filter, normalize

class ISRUCLoader(BaseDataset):
    def _load_data(self):
        """
        ISRUC subgroup I 데이터셋을 로드합니다.
        각 환자의 .edf 파일과 저녁 시간의 .hypnogram 파일을 읽습니다.
        """
        data_dict = {}
        labels_dict = {}
        
        subject_ids = range(1, 101) # 1 to 100
        
        for subj_id in subject_ids:
            # 파일 이름 형식: 1.edf, 1.hypnogram
            edf_path = os.path.join(self.base_path, f'{subj_id}.edf')
            hyp_path = os.path.join(self.base_path, f'{subj_id}.hypnogram')

            if os.path.exists(edf_path) and os.path.exists(hyp_path):
                raw = mne.io.read_raw_edf(edf_path, preload=True, verbose='ERROR')
                
                # 라벨 파일 읽기
                annotations = mne.read_annotations(hyp_path)
                raw.set_annotations(annotations, emit_warning=False)
                
                data_dict[subj_id] = raw
                # 라벨은 raw 객체 안에 저장됨
                labels_dict[subj_id] = None 
            else:
                print(f"Warning: Files not found for subject {subj_id}")
        
        return data_dict, labels_dict

    def _preprocess_data(self, raw, labels): # labels 인자는 사용되지 않음
        """
        필터링 후, 30초 에포크로 분할하고 시퀀스로 그룹화합니다.
        """
        params = self.props.preprocess_params
        
        # 채널 선택
        raw.pick_channels(self.props.channels)
        
        # 필터링
        raw_filtered = bandpass_filter(raw, params.l_freq, params.h_freq)
        
        # 30초 에포크 생성 및 라벨링
        events, event_id = mne.events_from_annotations(raw_filtered, chunk_duration=30.)
        
        # 라벨 매핑 (W, N1, N2, N3, R -> 0, 1, 2, 3, 4)
        # mne가 자동으로 event_id를 생성하지만, 순서를 보장하기 위해 명시적으로 매핑
        label_map = {'Sleep stage W': 0, 'Sleep stage N1': 1, 'Sleep stage N2': 2, 'Sleep stage N3': 3, 'Sleep stage R': 4}
        
        epochs = mne.Epochs(raw=raw_filtered, events=events, event_id=event_id, 
                            tmin=0., tmax=30., baseline=None, preload=True, on_missing='warn')

        # 라벨을 0-4 범위로 변환
        epoch_labels = np.array([label_map[desc] for desc in epochs.event_id if desc in label_map])
        
        data = epochs.get_data()
        data = normalize(data)
        
        # 시퀀스 데이터로 변환
        seq_len = self.props.get('sequence_length', 1)
        if seq_len > 1:
            num_seqs = len(data) // seq_len
            data = data[:num_seqs * seq_len].reshape(num_seqs, seq_len, *data.shape[1:])
            epoch_labels = epoch_labels[:num_seqs * seq_len].reshape(num_seqs, seq_len)
        
        return data, epoch_labels
