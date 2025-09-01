# -*- coding: utf-8 -*-
"""
CBraMod 모델의 공식적인 정의 파일.
범용 베이스 모델을 상속받아 구조를 완성합니다.
"""
from .base_model import EEGFoundationBaseModel

class CBraMod(EEGFoundationBaseModel):
    def __init__(self, config, **kwargs):
        """
        [수정] kwargs를 통해 추가적인 설정(pretrain_strategy_config)을
        베이스 모델에 전달할 수 있도록 합니다.
        """
        super().__init__(config, **kwargs)
        # CBraMod만을 위한 특별한 로직이 있다면 여기에 추가

