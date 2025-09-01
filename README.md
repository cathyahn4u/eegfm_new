# EEG Foundation Model Platform

  ## 1. 개요 (Overview)

  이 프로젝트는 EEG(뇌파) 데이터를 위한 유연하고 확장 가능한 Foundation Model 플랫폼입니다. CBraMod 논문 및 GitHub 리포지토리를 기반으로 리팩터링되었으며, 다양한 EEG 데이터셋과 모델 아키텍처를 쉽게 실험할 수 있는 모듈화된 구조를 제공합니다.

  핵심 목표는 연구자들이 데이터 전처리나 파이프라인 코드 작성에 들이는 시간을 최소화하고, 모델 아키텍처 설계 및 자가 학습(Self-supervised Learning) 전략 연구에 집중할 수 있도록 지원하는 것입니다.

  ## 2. 주요 특징 (Key Features)

  - **모듈화된 아키텍처**: 데이터 처리, 모델 컴포넌트, 학습 전략, 태스크 파이프라인이 명확하게 분리되어 있습니다.
  - **설정 파일 기반 제어 (`config-driven`)**: 모든 실험(사전 학습, 미세 조정, 평가)은 YAML 설정 파일을 통해 제어되므로, 코드 변경 없이 파라미터 튜닝 및 실험 관리가 용이합니다.
  - **유연한 모델 구성**: 베이스 모델 클래스를 상속받아 새로운 모델을 쉽게 추가할 수 있습니다. 모델의 모든 구성 요소(백본, 헤드 등)는 동적으로 조립됩니다.
  - **확장 가능한 자가 학습**: MAE, 대조 학습 등 다양한 자가 학습 전략을 모듈 형태로 추가하고 `config`에서 선택하여 사용할 수 있습니다.
  - **다중 데이터셋 지원**: 여러 공개 EEG 데이터셋(CHB-MIT, ISRUC, SEED 등)을 처리하기 위한 로더가 사전 구현되어 있으며, 새로운 데이터셋 추가가 용이합니다.
  - **재현성**: `config` 파일과 글로벌 시드 설정을 통해 실험 결과를 쉽게 재현할 수 있습니다.

  ## 3. 프로젝트 구조 (Project Structure)

  ```
  eeg_fm_platform/
  ├── configs/                  # 모든 YAML 설정 파일
  │   ├── default.yaml          # 기본 설정 (템플릿)
  │   ├── datasets.yaml         # 데이터셋별 상세 정보 (경로, 분할 등)
  │   └── experiments/
  │       └── finetune_chbmit.yaml # 실제 실행용 실험 설정 예시
  ├── src/                      # 소스 코드
  │   ├── data_handling/        # 데이터 로딩 및 전처리
  │   │   ├── datasets/         # 데이터셋별 로더 모듈
  │   │   ├── base_dataset.py   # 모든 데이터셋 로더의 부모 클래스
  │   │   └── data_loader.py    # 데이터셋 로더를 동적으로 선택하는 팩토리
  │   ├── learning_strategies/  # 자가 학습 전략 모듈 (e.g., MAE)
  │   ├── models/               # 모델 아키텍처
  │   │   ├── backbones/        # 모델의 핵심 특징 추출기
  │   │   ├── base_model.py     # 모든 모델의 부모 클래스
  │   │   ├── components.py     # 모델의 재사용 가능한 구성 요소
  │   │   └── cbramod.py        # CBraMod 모델 정의
  │   ├── tasks/                # 학습/평가 파이프라인
  │   │   ├── base_task.py      # 공통 학습/평가 로직
  │   │   ├── pretrain_task.py  # 사전 학습 파이프라인
  │   │   └── finetune_task.py  # 미세 조정 파이프라인
  │   └── utils/                # 헬퍼 함수 및 유틸리티
  ├── main.py                   # 프로젝트 메인 실행 파일
  ├── dummy_train_test.py       # 파이프라인 검증용 테스트 스크립트
  ├── requirements.txt          # 필요 라이브러리 목록
  └── setup_project.py          # 이 프로젝트를 생성하는 스크립트
  ```

  ## 4. 사용 방법 (Usage)

  ### 4.1. 환경 설정

  1.  프로젝트를 생성합니다. (이 스크립트를 실행)
  2.  필요한 라이브러리를 설치합니다.
  ```bash
  pip install -r requirements.txt
  ```
  3.  `configs/datasets.yaml` 파일을 열어 각 데이터셋의 실제 경로(`path`)를 자신의 환경에 맞게 수정합니다.

  ### 4.2. 파이프라인 테스트 (Dummy Data)

  실제 데이터를 다운로드하기 전에, 전체 파이프라인이 정상적으로 동작하는지 더미 데이터를 이용해 빠르게 확인할 수 있습니다.

  ```bash
  python dummy_train_test.py
  ```

  ### 4.3. 미세 조정 (Fine-tuning) 실행

  1.  `configs/experiments/` 디렉토리에 있는 실험 설정 파일 (예: `finetune_chbmit.yaml`)을 필요에 맞게 수정합니다. (사용할 이터셋, 배치 크기, 학습률 등)
  2.  `main.py`를 실행하여 미세 조정을 시작합니다.
  
      ```bash
      python main.py --config ./configs/experiments/finetune_chbmit.yaml
      ```

  ### 4.4. 사전 학습 (Pre-training) 실행

  1.  사전 학습용 실험 설정 파일을 생성합니다. (`mode: pretrain`으로 설정)
  2.  `main.py`를 실행하여 사전 학습을 시작합니다.

      ```bash
      python main.py --config path/to/your/pretrain_config.yaml
  ## 5. 확장 방법 (How to Extend)

  ### 5.1. 새로운 데이터셋 추가하기

  1.  `src/data_handling/datasets/` 디렉토리에 `new_dataset.py` 파일을 생성합니다.
  2.  `BaseDataset`을 상속받는 `NewDatasetLoader` 클래스를 만들고, `_load_data`와 `_preprocess_data` 메서드를 구현합니다.
  3.  `configs/datasets.yaml` 파일에 `NewDataset`의 정보(경로, 클래스 수, 피험자 분할 정보 등)를 추가합니다.
  4.  `data_loader.py`의 `DATASET_LOADERS` 딕셔너리에 `'NewDataset': NewDatasetLoader`를 추가합니다.

  ### 5.2. 새로운 모델 추가하기

  1.  `src/models/` 디렉토리에 `new_model.py` 파일을 생성합니다.
  2.  `EEGFoundationBaseModel`을 상속받는 `NewModel` 클래스를 정의합니다.
  3.  필요한 경우 `src/models/backbones`나 `src/models/components.py`에 새로운 구성 요소를 추가합니다.
  4.  `configs/default.yaml`의 `model` 섹션에 `NewModel`의 파라미터 설정을 추가합니다.
  5.  실험 설정 파일에서 `model_selection: 'NewModel'`로 변경하여 모델을 사용합니다.
