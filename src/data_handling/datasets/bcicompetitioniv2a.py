# -*- coding: utf-8 -*-
""" BCI Competition IV 2a 데이터셋 로더 """
import mne
import numpy as np
import glob
from tqdm import tqdm
from ..base_dataset import BaseDataset
from ..preprocessing import apply_filter

class BCICompetitionIV2aDataset(BaseDataset):
    def _load_data(self):
        all_X, all_y, all_groups = [], [], []
        
        # GDF 파일 경로 탐색
        subject_files = glob.glob(f"{self.props.path}/A*.gdf")
        
        for file_path in tqdm(subject_files, desc="Loading BCICIV2a"):
            subject_id = int(file_path.split('A0')[-1].split('T.gdf')[0])
            
            raw = mne.io.read_raw_gdf(file_path, preload=True, verbose=False)
            raw.drop_channels(['EOG-left', 'EOG-central', 'EOG-right'])
            
            # 이벤트(마커) 정보 추출
            events, event_id_dict = mne.events_from_annotations(raw, verbose=False)
            
            # 레이블 매핑
            labels_to_keep = self.props.label_mapping.keys()
            filtered_event_id = {k: v for k, v in event_id_dict.items() if k in labels_to_keep}
            
            # Epochs 생성 (이벤트 중심으로 데이터 자르기)
            epochs = mne.Epochs(raw, events, event_id=filtered_event_id, 
                                tmin=self.props.tmin, tmax=self.props.tmax, 
                                proj=False, baseline=None, preload=True, verbose=False)
            
            labels = epochs.events[:, -1]
            data = epochs.get_data() # (n_epochs, n_channels, n_times)
            
            # 전처리 적용
            data = apply_filter(data, fs=self.props.sampling_rate, lowcut=4.0, highcut=38.0)
            
            # 레이블을 config에 정의된 값으로 변환
            y_mapped = np.array([self.props.label_mapping[str(l)] for l in labels])
            
            all_X.append(data)
            all_y.append(y_mapped)
            all_groups.extend([subject_id] * len(data))
            
        return np.concatenate(all_X, axis=0), np.concatenate(all_y, axis=0), np.array(all_groups)

