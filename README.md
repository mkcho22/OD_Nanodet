# OD_Nanodet

NanoDet-Plus-m (416 × 416)을 이용한 `can` 단일 클래스 객체 탐지 학습 프로젝트입니다. YOLO TXT 라벨을 읽고 학습·검증 진행 상황과 평가 지표를 표로 출력합니다.

## GPU 기본값과 PC별 설정

| 파일 | 역할 | Git 추적 |
| --- | --- | --- |
| `requirements-gpu.txt` | 원격 GPU PC의 공통 설치 목록 | 포함 |
| `configs/nanodet-plus-m_416_can.gpu.yml` | 새 clone에서 사용하는 GPU 기본 설정 | 포함 |
| `requirements.txt` | 각 PC의 기존 설치 목록, 로컬 CPU 환경 포함 | 제외 |
| `configs/nanodet-plus-m_416_can.yml` | 각 PC에서 유지하는 개별 학습 설정 | 제외 |

`train.py`는 개별 설정인 `configs/nanodet-plus-m_416_can.yml`이 있으면 먼저 읽고, 없으면 저장소의 GPU 기본 설정을 읽습니다. 선택한 설정 파일과 학습 장치를 시작 표에 표시합니다.

기존 로컬 CPU 파일은 내용을 유지하며 사용하면 됩니다. 두 개별 파일은 `.gitignore`에 등록되어 있고 Git 추적에서도 제외되어, 일반적인 `git add .`와 push에 포함되지 않습니다. GPU 공통 기본 파일을 의도적으로 변경하면 해당 변경은 커밋할 수 있습니다.

데이터셋, 모델 가중치, 가상환경, 로그 및 학습 결과도 저장소에서 제외합니다.

## 원격 GPU PC 설치

대상 환경은 Windows의 NVIDIA GPU, Python 3.10, PyTorch `2.12.1+cu130`, torchvision `0.27.1+cu130`입니다. CUDA 13.0 빌드에 맞는 NVIDIA 드라이버가 필요합니다.

새로 clone하는 경우 Git Bash에서 실행합니다.

```bash
git clone https://github.com/mkcho22/OD_Nanodet.git
cd OD_Nanodet
py -3.10 -m venv .venv
source .venv/Scripts/activate
python -m pip install -r requirements-gpu.txt
python -m pip check
```

이미 `.venv`를 사용 중이면 새로 만들지 않고 활성화한 환경에서 설치 명령부터 실행하면 됩니다. 설치 명령은 반드시 저장소 루트에서 실행하세요.

`requirements-gpu.txt`는 다음 사항을 포함합니다.

- CUDA 13.0용 torch / torchvision 버전 고정
- `numpy==1.26.4`, `ml-dtypes==0.5.4` 조합으로 의존성 충돌 방지
- `-e ./nanodet_src`를 통해 로컬 NanoDet 패키지 설치
- 기존 PyTorch Lightning 1.9.5 및 학습 의존성 유지

소스에서도 삭제된 `torch._six` 의존성을 제거했습니다. Lightning 2.x로 변경하려면 기존 학습 훅에 대한 별도 코드 수정이 필요합니다.

설치와 GPU 인식을 확인합니다.

```bash
python -c "import torch, torchvision, nanodet; print('torch:', torch.__version__); print('torchvision:', torchvision.__version__); print('CUDA:', torch.cuda.is_available()); print('nanodet:', nanodet.__file__)"
```

GPU 환경에서는 torch `2.12.1+cu130`, torchvision `0.27.1+cu130`, `CUDA: True`가 나와야 합니다.

## 기존 원격 clone 업데이트

기존 clone에서 설정·requirements·소스를 직접 수정했다면 먼저 변경 내용을 확인하고 보관합니다.

```bash
git status
git stash push -m "before-gpu-defaults-update"
git pull --ff-only
python -m pip install -r requirements-gpu.txt
python -m pip check
```

기존 수정은 stash에 보관됩니다. 이번 버전은 공통 GPU 기본값과 호환성 수정을 포함하므로 stash를 자동으로 다시 적용하지 마세요. 개별 튜닝값이 있었다면 `git stash list`, `git stash show -p`로 확인하고 필요한 설정만 개별 YAML에 옮기세요.

이전 버전에서 추적하던 `requirements.txt`와 `configs/nanodet-plus-m_416_can.yml`은 이번 업데이트에서 추적이 종료됩니다. 다른 clone에서 pull하면 해당 파일이 제거될 수 있으므로 위 보관 단계를 먼저 수행하세요. 새 clone에는 두 파일이 없으며 GPU 기본값을 사용합니다.

## 로컬 CPU 테스트 유지

기존 CPU PC의 `requirements.txt`와 `configs/nanodet-plus-m_416_can.yml`은 그대로 유지합니다. 개별 YAML의 `device.gpu_ids: -1`이 CPU를 선택합니다.

기존 CPU 가상환경이 준비되어 있다면 다음처럼 실행하면 됩니다.

```bash
source .venv/Scripts/activate
python train.py
```

CPU PC에서는 GPU 전용 설치 파일을 실행할 필요가 없습니다. 현재 CPU 테스트 환경은 Python 3.10.11, PyTorch 1.13.1+cpu, torchvision 0.14.1+cpu입니다. 패키지를 다시 설치해야 하면 기존 개별 `requirements.txt`를 사용하고, NanoDet 패키지가 없다면 저장소 루트에서 `python -m pip install --no-deps -e ./nanodet_src`를 실행하세요.

원격 PC에서도 개별 설정이 필요하면 GPU 기본 파일을 복사한 뒤 수정할 수 있습니다.

```bash
cp configs/nanodet-plus-m_416_can.gpu.yml configs/nanodet-plus-m_416_can.yml
```

복사한 파일의 설정이 공통 GPU 기본값보다 우선합니다. 로컬 CPU 파일을 원격 PC에 복사하면 CPU 설정도 함께 적용되므로 시작 표의 설정 파일과 장치를 확인하세요.

## 데이터셋 준비

`datasets/can/train/`과 `datasets/can/val/`에 이미지와 동일한 이름의 YOLO TXT 라벨을 함께 둡니다.

```text
datasets/
└─ can/
   ├─ train/
   │  ├─ sample_001.jpg
   │  └─ sample_001.txt
   └─ val/
      ├─ sample_002.jpg
      └─ sample_002.txt
```

라벨 한 줄은 객체 하나를 뜻합니다.

```text
class_id x_center y_center width height
0 0.50 0.50 0.25 0.30
```

- 클래스 ID는 0부터 시작하며, 현재 `0`은 `can`입니다.
- 중심 좌표, 너비, 높이는 이미지 크기로 정규화한 값입니다.
- 한 이미지에 객체가 여러 개 있으면 줄을 추가합니다.
- 이미지와 TXT 파일은 같은 폴더에 두고 확장자를 제외한 파일명을 맞추세요.
- 클래스 이름은 YAML에 지정하며 별도의 `classes.txt`를 읽지 않습니다.

클래스를 바꾸려면 사용 중인 YAML의 `model.arch.head.num_classes`, `model.arch.aux_head.num_classes`, `data.train.class_names`, `data.val.class_names`, 최상위 `class_names`를 함께 수정하세요.

## 사전학습 가중치 준비

NanoDet 공식 [COCO 사전학습 체크포인트](https://github.com/RangiLyu/nanodet/releases/download/v1.0.0-alpha-1/nanodet-plus-m_416_checkpoint.ckpt)를 다음 위치에 저장합니다.

```text
weights/nanodet-plus-m_416_checkpoint.ckpt
```

Git Bash에서 다운로드하는 예시입니다.

```bash
mkdir -p weights
curl -L --fail https://github.com/RangiLyu/nanodet/releases/download/v1.0.0-alpha-1/nanodet-plus-m_416_checkpoint.ckpt -o weights/nanodet-plus-m_416_checkpoint.ckpt
```

현재 학습 코드는 이 체크포인트가 없으면 오류를 발생시킵니다.

## 학습 실행

데이터와 가중치를 준비한 뒤 저장소 루트에서 실행합니다.

```bash
python train.py
```

`configs/nanodet-plus-m_416.yml`은 원본 COCO 설정 참고용입니다.

### 현재 기본값: 짧은 실행 확인

GPU 기본 설정은 장치만 GPU로 지정하며 기존 학습 길이와 배치 제한은 유지합니다.

| 항목 | GPU 기본값 |
| --- | --- |
| 장치 | 첫 번째 NVIDIA GPU (`gpu_ids: [0]`) |
| 입력 크기 | 416 × 416 |
| 배치 크기 | 4 |
| 데이터 로더 작업자 | 0 |
| 최대 epoch | 2 |
| epoch당 학습 배치 | 최대 20 |
| epoch당 검증 배치 | 최대 5 |

현재 검증 지표는 제한된 검증 배치에서 계산됩니다. 전체 검증 데이터의 성능으로 해석하지 마세요.

### 전체 데이터로 학습하기

1. `train.py`의 `pl.Trainer(...)`에서 `limit_train_batches=1.0`, `limit_val_batches=1.0`으로 변경하거나 두 인자를 제거합니다. `1.0`은 전체 배치를 뜻하고 정수 `1`은 한 배치만 실행합니다.
2. 사용 중인 YAML의 `schedule.total_epochs`와 `schedule.lr_schedule.T_max`를 계획한 학습 길이에 맞춥니다. 현재는 `total_epochs: 2`, `T_max: 300`입니다.
3. 장치와 메모리에 맞춰 `device.batchsize_per_gpu` 등을 조정합니다. PC별 튜닝값은 Git에서 제외되는 개별 YAML에 보관하세요.

조기 종료는 `val_mAP`를 감시하며 `patience=10`, `min_delta=0.001`입니다.

## 학습 결과

기본 저장 위치는 `workspace/nanodet-plus-m_416_can/`입니다.

- 마지막 체크포인트: `model_last.ckpt`
- 최적 체크포인트: `model_best/model_best.ckpt`
- 최적 모델 가중치: `model_best/nanodet_model_best.pth`
- 로그 및 평가 결과: 같은 작업 폴더 하위

## 원본 프로젝트 및 라이선스

이 프로젝트는 [RangiLyu/nanodet](https://github.com/RangiLyu/nanodet)를 기반으로 합니다. `nanodet_src/`에는 이 프로젝트의 학습 코드와 호환성 수정이 포함되어 있습니다.

원본 저장소의 기준 커밋: `be9b4a9001d7f9b6fc89c2df31ae8d428e35b4f0`

원본 라이선스는 [nanodet_src/LICENSE](nanodet_src/LICENSE)이며, 파일별 저작권 표시와 하위 구성 요소의 라이선스를 함께 유지합니다. 원본 사용법은 [nanodet_src/README.md](nanodet_src/README.md)를 참고하세요.
