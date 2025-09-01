# -*- coding: utf-8 -*-
"""
chbmit.py
[아키텍처 개선] 복잡한 채널 처리 로직을 제거하고,
부모 클래스의 공용 메서드인 self._apply_montage(raw)를 호출하도록 수정하여
코드를 대폭 단순화합니다.
"""
import mne
import numpy as np
import os
import re
from ..base_dataset import BaseDataset
from ..preprocessing import resample, create_epochs, normalize

class CHBMITLoader(BaseDataset):
    def _load_data(self):
        """
        CHB-MIT 데이터셋을 로드합니다.
        각 환자(.edf) 파일과 발작 시간 정보(.seizures)를 읽습니다.
        """
        data_dict = {}
        seizure_info_dict = {}
        
        subject_folders = sorted([d for d in os.listdir(self.base_path) if d.startswith('chb') and os.path.isdir(os.path.join(self.base_path, d))])

        for folder in subject_folders:
            subj_id = int(re.search(r'\d+', folder).group())
            subj_path = os.path.join(self.base_path, folder)
            
            summary_file = os.path.join(subj_path, f'{folder}-summary.txt')
            if not os.path.exists(summary_file):
                continue

            seizures = self._parse_summary(summary_file)
            seizure_info_dict[subj_id] = seizures

            raw_list = []
            edf_files = sorted([f for f in os.listdir(subj_path) if f.endswith('.edf')])
            for edf_file in edf_files:
                try:
                    raw = mne.io.read_raw_edf(os.path.join(subj_path, edf_file), preload=True, verbose='ERROR')
                    raw_list.append(raw)
                except Exception as e:
                    print(f"Warning: Could not read {edf_file}. Error: {e}")
            
            if raw_list:
                data_dict[subj_id] = mne.concatenate_raws(raw_list)
        
        return data_dict, seizure_info_dict
        
    def _parse_summary(self, file_path):
        """-summary.txt 파일에서 발작 시작/종료 시간을 파싱합니다."""
        seizures = []
        with open(file_path, 'r') as f:
            content = f.read()
            file_blocks = re.split(r'File Name:', content)[1:]
            for block in file_blocks:
                lines = block.strip().split('\n')
                file_name = lines[0].strip()
                
                num_seizures_line = next((line for line in lines if "Number of Seizures" in line), None)
                if num_seizures_line and int(num_seizures_line.split(':')[-1].strip()) > 0:
                    starts = [l for l in lines if "Seizure Start Time" in l]
                    ends = [l for l in lines if "Seizure End Time" in l]
                    for start, end in zip(starts, ends):
                        start_sec = int(start.split(':')[-1].strip().split()[0])
                        end_sec = int(end.split(':')[-1].strip().split()[0])
                        seizures.append({'file_name': file_name, 'start': start_sec, 'end': end_sec})
        return seizures

    def _preprocess_data(self, raw, seizure_info):
        """
        Bipolar montage 적용, 에포크 생성 및 라벨링을 수행합니다.
        """
        # --- 아키텍처 개선: 공용 몽타주 함수 호출 ---
        try:
            raw_montaged = self._apply_montage(raw)
        except ValueError as e:
            print(f"Warning: Skipping subject due to montage error. {e}")
            return np.array([]), np.array([])
            
        raw_resampled = resample(raw_montaged, self.props['preprocess_params']['resample_freq'])

        epochs = create_epochs(raw_resampled, duration=self.props['duration_secs'])
        
        labels = np.zeros(len(epochs))
        for seizure in seizure_info:
             seizure_start_epoch = seizure['start'] // self.props['duration_secs']
             seizure_end_epoch = seizure['end'] // self.props['duration_secs']
             if seizure_end_epoch < len(labels):
                 labels[seizure_start_epoch : seizure_end_epoch + 1] = 1

        data = epochs.get_data()
        data = normalize(data)
        
        return data, labels


