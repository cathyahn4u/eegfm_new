# -*- coding: utf-8 -*-
"""
helpers.py
[아키텍처 최종 수정] 설정 로딩 로직을 근본적으로 수정합니다.
이제 load_config 함수는 모든 YAML 파일을 명확한 우선순위에 따라 병합한
'최종 딕셔너리'를 먼저 생성한 뒤, 이 딕셔너리를 사용하여 dataclass 객체를
생성합니다. 이를 통해 default.yaml의 값이 config_schema.py의 기본값을
항상 올바르게 오버라이드하도록 보장하고, 설정 관련 혼동을 원천적으로 차단합니다.
"""
import yaml
from pathlib import Path
from typing import Type, TypeVar, Dict, Any
from dataclasses import is_dataclass, fields
from src.config_schema import AppConfig

T = TypeVar('T')

def _recursive_update(base_dict: Dict, new_dict: Dict) -> Dict:
    """두 딕셔너리를 재귀적으로 병합합니다. new_dict의 값이 base_dict의 값을 덮어씁니다."""
    for key, value in new_dict.items():
        if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
            base_dict[key] = _recursive_update(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict

def _from_dict_to_dataclass(cls: Type[T], data: Dict[str, Any]) -> T:
    """딕셔너리를 데이터클래스 객체로 재귀적으로 변환합니다."""
    # 데이터클래스 필드 정보를 가져옵니다.
    field_types = {f.name: f.type for f in fields(cls)}
    constructor_args = {}

    # 딕셔너리의 각 항목에 대해
    for name, value in data.items():
        # 데이터클래스에 정의된 필드인 경우에만 처리
        if name in field_types:
            field_type = field_types[name]
            
            # 필드 타입이 또 다른 데이터클래스이고, 값도 딕셔너리이면 재귀적으로 변환
            # is_dataclass는 type 객체에만 작동하므로 field_type을 사용합니다.
            if is_dataclass(field_type) and isinstance(value, dict):
                constructor_args[name] = _from_dict_to_dataclass(field_type, value)
            else:
                constructor_args[name] = value
    
    # 최종적으로 구성된 인자로 데이터클래스 인스턴스 생성
    # constructor_args에 없는 필드는 dataclass의 기본값을 사용하게 됩니다.
    return cls(**constructor_args)

def load_config(experiment_config_path: str) -> AppConfig:
    """
    정의된 우선순위에 따라 모든 설정 파일을 병합하고, 최종 AppConfig 객체를 생성합니다.
    우선순위: 실험 설정 > default.yaml > datasets.yaml > 스키마 기본값
    """
    # 1. 기본 경로 설정
    configs_dir = Path(__file__).resolve().parent.parent.parent / 'configs'
    default_config_path = configs_dir / 'default.yaml'
    datasets_config_path = configs_dir / 'datasets.yaml'

    # 2. 모든 YAML 파일을 순서대로 로드하여 하나의 딕셔너리로 병합
    final_config_dict = {}

    # 2.1. 기본 설정(default.yaml)을 가장 먼저 로드
    with open(default_config_path, 'r', encoding='utf-8') as f:
        default_config = yaml.safe_load(f)
    final_config_dict = _recursive_update(final_config_dict, default_config)

    # 2.2. 데이터셋 설정(datasets.yaml)을 그 위에 덮어씀
    if datasets_config_path.exists():
        with open(datasets_config_path, 'r', encoding='utf-8') as f:
            datasets_config = yaml.safe_load(f)
        final_config_dict = _recursive_update(final_config_dict, datasets_config)

    # 2.3. 마지막으로, 실험별 설정(experiment_config_path)을 가장 높은 우선순위로 덮어씀
    with open(experiment_config_path, 'r', encoding='utf-8') as f:
        experiment_config = yaml.safe_load(f)
    final_config_dict = _recursive_update(final_config_dict, experiment_config)

    # 3. 최종적으로 병합된 딕셔너리를 사용하여 AppConfig 데이터클래스 객체 생성
    return _from_dict_to_dataclass(AppConfig, final_config_dict)

def resolve_paths(config: AppConfig) -> AppConfig:
    """설정 객체의 경로 문자열에 있는 플레이스홀더를 실제 값으로 변환합니다."""
    replacements = {
        "{dataset_name}": config.dataset_name,
        "{model_selection}": config.model_selection,
    }

    def _resolve_str(s: str) -> str:
        if not s: return s
        for placeholder, value in replacements.items():
            s = s.replace(placeholder, str(value))
        return s

    # 재귀적으로 모든 문자열 경로를 변환
    def _recursive_resolve(obj):
        if isinstance(obj, str):
            return _resolve_str(obj)
        if isinstance(obj, dict):
            return {k: _recursive_resolve(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_recursive_resolve(i) for i in obj]
        if is_dataclass(obj):
            for f in fields(obj):
                setattr(obj, f.name, _recursive_resolve(getattr(obj, f.name)))
        return obj

    return _recursive_resolve(config)

