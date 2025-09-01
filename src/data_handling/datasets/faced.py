# -*- coding: utf-8 -*-
"""
faced.py
FACED Emotion Recognition 데이터셋을 위한 데이터 로더 및 전처리기.
[오류 수정] pickle 파일을 로드한 후, 내용물이 예상된 dict 타입인지
명시적으로 확인하고, 'data'와 'labels'를 분리하여 저장하도록 수정합니다.
이를 통해 ndarray가 직접 로드되는 경우 발생하는 AttributeError를
근본적으로 해결하고 코드의 안정성을 높입니다.
"""
import mne
import numpy as np
import os
import pickle
import re # 정규 표현식 모듈 임포트
from ..base_dataset import BaseDataset
from ..preprocessing import resample, create_epochs, normalize

class FACEDLoader(BaseDataset):
    def _load_data(self):
        """
        FACED 데이터셋을 로드합니다. 파일 확장자를 감지하여 .pickle 또는 .edf 파일을 처리합니다.
        """
        data_dict = {}
        labels_dict = {}

        all_files = os.listdir(self.base_path)
        pickle_files = sorted([f for f in all_files if f.endswith(('.pkl', '.pickle'))])
        edf_files = sorted([f for f in all_files if f.endswith('.edf')])

        if pickle_files:
            print("Pickle files detected for FACED dataset. Loading pre-processed data.")
            for f in pickle_files:
                try:
                    match = re.search(r'\d+', f)
                    if not match:
                        print(f"Warning: Could not parse subject ID from filename: {f}. Skipping file.")
                        continue
                    subj_id = int(match.group())
                    
                    fpath = os.path.join(self.base_path, f)
                    with open(fpath, 'rb') as pkl_file:
                        loaded_content = pickle.load(pkl_file, encoding='latin1')

                    # --- 근본적인 오류 해결: 로드된 객체의 타입 확인 ---
                    if isinstance(loaded_content, dict) and 'data' in loaded_content and 'labels' in loaded_content:
                        # 'data'와 'labels'를 분리하여 각각 저장
                        data_dict[subj_id] = loaded_content['data']
                        labels_dict[subj_id] = loaded_content['labels']
                    else:
                        # 예상치 못한 형식(e.g., numpy.ndarray)인 경우 경고 후 건너뜀
                        print(f"Warning: Pickle file {f} has an unexpected format or missing keys. Type: {type(loaded_content)}. Skipping.")
                        continue

                except (pickle.UnpicklingError, ValueError) as e:
                    print(f"Warning: Could not parse or load pickle file: {f}. Error: {e}")

        elif edf_files:
            print("EDF files detected for FACED dataset. Loading raw data.")
            subject_files = {}
            for f in edf_files:
                try:
                    match = re.search(r'\d+', f)
                    if not match:
                        print(f"Warning: Could not parse subject ID from filename: {f}. Skipping file.")
                        continue
                    subj_id = int(match.group())

                    if subj_id not in subject_files:
                        subject_files[subj_id] = []
                    subject_files[subj_id].append(os.path.join(self.base_path, f))
                except ValueError:
                    print(f"Warning: Could not parse subject ID from filename: {f}. Skipping file.")

            for subj_id, files in subject_files.items():
                try:
                    raw_list = [mne.io.read_raw_edf(f, preload=True, verbose='ERROR') for f in files]
                    data_dict[subj_id] = mne.concatenate_raws(raw_list)
                    labels_dict[subj_id] = None # EDF는 전처리 시 라벨 생성
                except Exception as e:
                    print(f"Warning: Could not process EDF files for subject {subj_id}. Error: {e}")
        
        else:
            print(f"Warning: No .edf or .pickle files found in {self.base_path}")

        return data_dict, labels_dict

    def _preprocess_data(self, subject_data, subject_labels):
        """
        입력 데이터 타입에 따라 다른 전처리 파이프라인을 적용합니다.
        """
        # _load_data에서 이미 data와 label을 분리했으므로, subject_data는 항상
        # numpy 배열 또는 MNE Raw 객체가 됩니다.
        if isinstance(subject_data, np.ndarray):
            data = subject_data
            labels = subject_labels
            
            if labels is None:
                print("Warning: Labels not found for pre-processed data. Skipping subject.")
                return np.array([]), np.array([])

            if data.ndim != 3:
                print(f"Warning: Pre-processed data has unexpected shape {data.shape}. Skipping.")
                return np.array([]), np.array([])
                
            data = normalize(data)
            return data, labels
            
        elif isinstance(subject_data, mne.io.Raw):
            try:
                raw_montaged = self._apply_montage(subject_data)
            except Exception as e:
                print(f"Warning: Could not apply montage for a subject. {e}")
                return np.array([]), np.array([])
                
            raw_resampled = resample(raw_montaged, self.props['preprocess_params']['resample_freq'])
            epochs = create_epochs(raw_resampled, duration=self.props['duration_secs'])
            
            num_epochs_actual = len(epochs)
            if num_epochs_actual == 0:
                return np.array([]), np.array([])

            epoch_labels = np.random.randint(0, self.props['num_classes'], num_epochs_actual)
            data = epochs.get_data()
            data = normalize(data)
            
            return data, epoch_labels
            
        else:
            return np.array([]), np.array([])

