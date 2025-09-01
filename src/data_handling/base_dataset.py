# -*- coding: utf-8 -*-
"""
base_dataset.py
[아키텍처 개선] 채널 처리를 위한 공용 메서드 '_apply_montage'를 추가합니다.
이 메서드는 config의 montage.type에 따라 적절한 MNE 함수를 동적으로
호출하여, 개별 데이터 로더의 코드를 단순화하고 재사용성을 높입니다.
"""
from abc import ABC, abstractmethod
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import mne
from src.config_schema import AppConfig

class BaseDataset(ABC):
    """모든 데이터셋 로더의 추상 베이스 클래스."""
    def __init__(self, config: AppConfig):
        self.config = config
        self.dataset_name = config.dataset_name
        self.props = config.datasets['properties'][self.dataset_name]
        self.base_path = os.path.expanduser(self.props['path'])

    @abstractmethod
    def _load_data(self):
        """
        데이터 파일로부터 원시 데이터를 로드합니다.
        반환값: (피험자별 데이터 딕셔너리, 피험자별 라벨 딕셔너리)
        """
        pass

    @abstractmethod
    def _preprocess_data(self, subject_data, subject_labels):
        """
        단일 피험자의 데이터를 전처리합니다.
        반환값: (전처리된 데이터 배열, 라벨 배열)
        """
        pass
    
    def _apply_montage(self, raw: mne.io.Raw) -> mne.io.Raw:
        """
        Config에 정의된 몽타주 설정을 MNE Raw 객체에 적용합니다.
        
        Args:
            raw (mne.io.Raw): 원본 Raw 객체.

        Returns:
            mne.io.Raw: 몽타주가 적용된 Raw 객체.
        """
        montage_config = self.props['montage']
        montage_type = montage_config['type']
        channels = montage_config['channels']
        
        # 채널 이름 소문자로 통일 (파일마다 대소문자가 다른 경우가 많음)
        raw.rename_channels(lambda x: x.lower())
        
        if montage_type == 'pick':
            # 필요한 채널만 선택
            channels_lower = [ch.lower() for ch in channels]
            # Raw 객체에 존재하는 채널만 선택
            channels_to_pick = [ch for ch in channels_lower if ch in raw.ch_names]
            if len(channels_to_pick) < len(channels_lower):
                 print(f"Warning: Not all channels found. Found {len(channels_to_pick)}/{len(channels_lower)}.")
            raw.pick_channels(channels_to_pick)

        elif montage_type == 'bipolar':
            # Bipolar montage 적용
            anodes, cathodes = zip(*(ch.lower().split('-') for ch in channels))
            
            if all(ch in raw.ch_names for ch in anodes) and all(ch in raw.ch_names for ch in cathodes):
                raw = mne.set_bipolar_reference(raw, anode=list(anodes), cathode=list(cathodes), ch_name=channels, copy=True)
            else:
                raise ValueError("Could not create bipolar montage. Not all required channels are present in the raw data.")
        
        else:
            raise ValueError(f"Unsupported montage type: {montage_type}")
            
        return raw

    def get_dataloaders(self):
        """피험자 단위로 데이터를 분할하고 PyTorch DataLoader를 생성합니다."""
        all_subjects_data, all_subjects_labels = self._load_data()
        
        split_info = self.props['split_info']
        
        def get_data_for_subjects(subject_ids):
            # subject_ids가 범위(e.g., [1, 80])로 주어졌을 경우 실제 ID 리스트 생성
            if isinstance(subject_ids, list) and len(subject_ids) == 2:
                subject_ids = range(subject_ids[0], subject_ids[1] + 1)

            data_list, labels_list = [], []
            for subj_id in subject_ids:
                if subj_id in all_subjects_data:
                    data, labels = self._preprocess_data(all_subjects_data[subj_id], all_subjects_labels[subj_id])
                    if data.size > 0: # 데이터가 있는 경우에만 추가
                        data_list.append(data)
                        labels_list.append(labels)
            
            if not data_list:
                return np.array([]), np.array([])
            
            return np.concatenate(data_list, axis=0), np.concatenate(labels_list, axis=0)

        X_train, y_train = get_data_for_subjects(split_info['train'])
        X_val, y_val = get_data_for_subjects(split_info['val'])
        X_test, y_test = get_data_for_subjects(split_info['test'])

        print(f"Dataset '{self.dataset_name}' loaded and split.")
        print(f"Train samples: {len(y_train)}, Val samples: {len(y_val)}, Test samples: {len(y_test)}")

        train_dataset = EEGDataset(X_train, y_train)
        val_dataset = EEGDataset(X_val, y_val)
        test_dataset = EEGDataset(X_test, y_test)

        batch_size = self.config.training.finetune.batch_size
        num_workers = self.config.data_handling.num_workers
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        return train_loader, val_loader, test_loader

class EEGDataset(Dataset):
    """간단한 PyTorch Dataset 클래스."""
    def __init__(self, data, labels):
        self.data = data.astype(np.float32)
        self.labels = labels.astype(np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

