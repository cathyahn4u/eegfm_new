# -*- coding: utf-8 -*-
"""
data_loader.py
[기능 추가] 모든 데이터셋 로더 클래스를 동적으로 임포트하고
선택할 수 있도록 DATASET_LOADERS 딕셔너리를 확장합니다.
"""
import importlib

# 여기에 새로운 데이터셋 로더를 추가합니다.
# 키는 datasets.yaml의 데이터셋 이름과 일치해야 합니다.
# 값은 '모듈이름.클래스이름' 형식입니다.
DATASET_LOADERS = {
    'BCICompetitionIV2a': 'bcicompetitioniv2a.BCICIV2aLoader',
    'SEED': 'seed.SEEDLoader',
    'CHBMIT': 'chbmit.CHBMITLoader',
    'ISRUC': 'isruc.ISRUCLoader',
    'FACED': 'faced.FACEDLoader',
    'PhysioNetMI': 'physionetmi.PhysioNetMILoader',
    'Mumtaz': 'mumtaz.MumtazLoader',
    'MentalArithmetic': 'mentalarithmetic.MentalArithmeticLoader',
    'TUAB': 'tuab.TUABLoader',
    'TUEV': 'tuev.TUEVLoader',
    'TUEG': 'tueg.TUEGLoader', # <-- TUEG 로더 등록    
}

def get_dataloaders(config):
    """
    설정 파일에 명시된 데이터셋 이름에 따라 적절한 로더를 선택하여
    학습, 검증, 테스트 데이터로더를 반환하는 팩토리 함수입니다.
    """
    dataset_name = config.dataset_name

    if dataset_name not in DATASET_LOADERS:
        raise ValueError(f"Dataset '{dataset_name}' is not supported. Please add it to DATASET_LOADERS in data_loader.py")

    # 모듈 경로와 클래스 이름을 분리
    module_path, class_name = DATASET_LOADERS[dataset_name].rsplit('.', 1)
    
    try:
        # 데이터셋 로더 모듈을 동적으로 임포트
        DatasetModule = importlib.import_module(f"src.data_handling.datasets.{module_path}")
        # 모듈에서 클래스를 가져옴
        DatasetLoaderClass = getattr(DatasetModule, class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Could not import {class_name} from src.data_handling.datasets.{module_path}. Error: {e}")
    
    # 데이터셋 로더 인스턴스 생성 및 데이터로더 반환
    dataset_loader = DatasetLoaderClass(config)
    return dataset_loader.get_dataloaders()

