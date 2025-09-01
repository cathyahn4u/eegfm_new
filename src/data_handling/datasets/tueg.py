# -*- coding: utf-8 -*-
"""
tueg.py
Temple University Hospital EEG Corpus (TUEG)를 위한 데이터 로더 및 전처리기.
대규모 비지도 사전 학습을 위한 데이터를 처리합니다.
"""
import mne
import numpy as np
import os
from sklearn.model_selection import train_test_split
from ..base_dataset import BaseDataset
from ..preprocessing import resample, create_epochs, normalize, bandpass_filter

class TUEGLoader(BaseDataset):
    """TUEG 데이터셋 로더 클래스."""

    def get_dataloaders(self):
        """TUEG에 특화된 데이터로더 생성 로직을 오버라이드합니다."""
        all_filepaths = self._find_all_files()
        
        # 파일 목록을 기준으로 학습/검증 세트 분할
        train_files, val_files = train_test_split(
            all_filepaths,
            test_size=self.props['split_info']['val_ratio'],
            random_state=self.config.system.seed
        )

        print(f"Found {len(all_filepaths)} total files for TUEG.")
        print(f"Splitting into {len(train_files)} training files and {len(val_files)} validation files.")

        train_dataset = TUEGDataset(train_files, self.config)
        val_dataset = TUEGDataset(val_files, self.config)

        # 사전 학습 시에는 배치 크기가 다를 수 있으므로 config에서 직접 가져옴
        batch_size = self.config.training.pretrain.batch_size
        num_workers = self.config.data_handling.num_workers

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=self._collate_fn)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=self._collate_fn)

        # 사전 학습에서는 테스트 로더가 필요 없음
        return train_loader, val_loader, None

    def _find_all_files(self):
        """베이스 경로에서 모든 .edf 파일을 재귀적으로 찾습니다."""
        filepaths = []
        for root, _, files in os.walk(self.base_path):
            for f in files:
                if f.endswith('.edf'):
                    filepaths.append(os.path.join(root, f))
        return filepaths
        
    def _collate_fn(self, batch):
        """배치에서 None 값을 제거하기 위한 콜레이트 함수."""
        batch = [b for b in batch if b is not None]
        if not batch:
            return None, None # 배치가 비어있는 경우
        
        data, labels = zip(*batch)
        return torch.from_numpy(np.concatenate(data, axis=0)), torch.from_numpy(np.concatenate(labels, axis=0))


class TUEGDataset(torch.utils.data.Dataset):
    """
    TUEG를 위한 PyTorch Dataset 클래스.
    파일을 하나씩 읽고 처리하여 메모리 문제를 방지합니다.
    """
    def __init__(self, filepaths, config):
        self.filepaths = filepaths
        self.config = config
        self.props = config.datasets['properties']['TUEG']
        self.params = self.props['preprocess_params']

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        fpath = self.filepaths[idx]
        try:
            raw = mne.io.read_raw_edf(fpath, preload=True, verbose='ERROR')
            
            # 1. 너무 짧은 레코딩 제거
            if raw.times[-1] < self.params['min_duration_mins'] * 60:
                return None
            
            # 2. 처음과 끝 1분씩 제거
            raw.crop(tmin=self.params['crop_mins'] * 60, tmax=raw.times[-1] - self.params['crop_mins'] * 60)

            # 3. 몽타주 적용 (채널 선택)
            montage_config = self.props['montage']
            channels_lower = [ch.lower() for ch in montage_config['channels']]
            raw.rename_channels(lambda x: x.lower())
            
            # TCP 채널 이름 형식(e.g., 'eeg fp1-ref')을 표준 형식('fp1')으로 변환
            ch_map = {ch: ch.split(' ')[1].split('-')[0] for ch in raw.ch_names if ' ' in ch}
            raw.rename_channels(ch_map)
            
            channels_to_pick = [ch for ch in channels_lower if ch in raw.ch_names]
            if len(channels_to_pick) < len(channels_lower):
                return None # 필요한 모든 채널이 없는 경우 이 파일을 건너뜀
            raw.pick_channels(channels_to_pick)

            # 4. 필터링 및 리샘플링
            raw_filtered = bandpass_filter(raw, self.params['l_freq'], self.params['h_freq'])
            raw_filtered.notch_filter(freqs=self.params['notch_freq'])
            raw_resampled = resample(raw_filtered, self.params['resample_freq'])
            
            # 5. 에포크 생성
            epochs = create_epochs(raw_resampled, duration=self.props['duration_secs'])
            data = epochs.get_data() # (n_epochs, n_channels, n_times)
            
            # 6. 불량 샘플 제거 (진폭 기준)
            threshold = self.params['amplitude_threshold_uv'] * 1e-6 # V 단위로 변환
            valid_indices = ~np.any(np.abs(data) > threshold, axis=(1, 2))
            if not np.any(valid_indices):
                return None # 유효한 에포크가 없는 경우
            data = data[valid_indices]

            # 7. 정규화 (LaBraM 방식)
            data /= 100e-6
            data = np.clip(data, -1, 1)

            # 사전 학습에서는 라벨이 필요 없으므로 더미 라벨 생성
            labels = np.zeros(len(data), dtype=np.int64)

            return data.astype(np.float32), labels

        except Exception as e:
            print(f"Warning: Skipping file {fpath} due to error: {e}")
            return None
