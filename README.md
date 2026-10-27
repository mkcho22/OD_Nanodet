# OD_Nanodet

NanoDet-Plus-m (416 × 416)을 이용한 `can` 단일 클래스 객체 탐지 학습 프로젝트입니다. YOLO TXT 라벨을 읽어 학습하며, 학습·검증 진행 상황과 평가 지표를 표로 출력합니다.

## 저장소 구성

```text
OD_Nanodet/
├─ configs/                 # 기본 COCO 설정과 can 학습 설정
├─ nanodet_src/             # NanoDet 소스 및 원본 LICENSE
├─ train.py                 # 이 프로젝트의 학습 진입점
├─ requirements.txt         # Python 의존성
├─ README.md
└─ .gitignore
```

데이터셋, 가중치, 가상환경, 로그 및 학습 결과는 저장소에 포함하지 않습니다. 아래 절차에 따라 데이터를 준비하고 사전학습 가중치를 내려받아야 합니다.

## 실행 환경 및 설치

확인한 환경은 Windows, Python 3.10.11, PyTorch 1.13.1+cpu, torchvision 0.14.1+cpu, PyTorch Lightning 1.9.5입니다. 기존 NanoDet 코드가 `torch._six`를 사용하므로 PyTorch 2.x로 임의 업그레이드하지 마세요.

저장소 루트에서 PowerShell로 실행합니다. 다음 명령은 CPU 환경을 구성합니다.

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install "torch==1.13.1+cpu" "torchvision==0.14.1+cpu" --extra-index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install --no-deps -e ./nanodet_src
```

`nanodet_src`는 설치해야 하는 로컬 Python 패키지입니다. 폴더를 복사하는 것만으로 `from nanodet ...`를 사용할 수 있는 것은 아닙니다.

GPU로 학습하려면 PyTorch 1.13.1 / torchvision 0.14.1에 맞는 CUDA 빌드를 설치하고, 학습 설정의 `device.gpu_ids`를 `[0]`으로 바꾸세요. 이 저장소의 기본값은 CPU이며 CUDA 실행은 별도로 검증하지 않았습니다.

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
- 별도의 `classes.txt`는 이 학습 코드가 읽지 않습니다. 클래스 이름은 YAML에 지정합니다.

클래스를 변경할 때는 `configs/nanodet-plus-m_416_can.yml`의 `model.arch.head.num_classes`, `model.arch.aux_head.num_classes`, `data.train.class_names`, `data.val.class_names`, 최상위 `class_names`를 함께 수정하세요.

## 사전학습 가중치 준비

NanoDet 공식 [COCO 사전학습 체크포인트](https://github.com/RangiLyu/nanodet/releases/download/v1.0.0-alpha-1/nanodet-plus-m_416_checkpoint.ckpt)를 다음 위치에 저장합니다.

```text
weights/nanodet-plus-m_416_checkpoint.ckpt
```

PowerShell 다운로드 예시:

```powershell
New-Item -ItemType Directory -Force weights | Out-Null
Invoke-WebRequest -Uri "https://github.com/RangiLyu/nanodet/releases/download/v1.0.0-alpha-1/nanodet-plus-m_416_checkpoint.ckpt" -OutFile "weights/nanodet-plus-m_416_checkpoint.ckpt"
```

현재 학습 코드는 이 체크포인트가 없으면 오류를 발생시킵니다.

## 학습 실행

저장소 루트에서 실행합니다.

```powershell
.\.venv\Scripts\python.exe train.py
```

`train.py`는 `configs/nanodet-plus-m_416_can.yml`을 읽습니다. 별도의 `--config` 인자는 없습니다. `configs/nanodet-plus-m_416.yml`은 원본 COCO 설정 참고용입니다.

### 현재 기본값: 짧은 실행 확인

| 항목 | 값 |
| --- | --- |
| 장치 | CPU (`gpu_ids: -1`) |
| 입력 크기 | 416 × 416 |
| 배치 크기 | 4 |
| 데이터 로더 작업자 | 0 |
| 최대 epoch | 2 |
| epoch당 학습 배치 | 최대 20 |
| epoch당 검증 배치 | 최대 5 |

현재 설정의 검증 지표는 제한된 검증 배치에서 계산됩니다. 전체 검증 데이터의 성능으로 해석하지 마세요.

### 전체 데이터로 학습하기

1. `train.py`의 `pl.Trainer(...)`에서 아래 두 항목을 변경하거나 제거합니다.
   ```python
   limit_train_batches=1.0,
   limit_val_batches=1.0,
   ```
   `1.0`은 전체 배치를 뜻하고, 정수 `1`은 한 배치만 실행합니다.
2. YAML의 `schedule.total_epochs`와 `schedule.lr_schedule.T_max`를 계획한 학습 길이에 맞춥니다. 예를 들어 둘 다 `300`으로 설정할 수 있습니다. 현재 파일은 `total_epochs: 2`, `T_max: 300`입니다.
3. 장치와 메모리에 맞춰 `device.gpu_ids`, `batchsize_per_gpu`를 설정합니다.

조기 종료는 `val_mAP`를 감시하며 `patience=10`, `min_delta=0.001`입니다.

## 학습 결과

기본 저장 위치는 `workspace/nanodet-plus-m_416_can/`입니다.

- 마지막 체크포인트: `model_last.ckpt`
- 최적 체크포인트: `model_best/model_best.ckpt`
- 최적 모델 가중치: `model_best/nanodet_model_best.pth`
- 로그 및 평가 결과: 같은 작업 폴더 하위

이 경로의 파일은 Git에서 제외됩니다. 학습 결과를 공유할 때는 별도 다운로드 위치를 마련하세요.

## 원본 프로젝트 및 라이선스

이 프로젝트는 [RangiLyu/nanodet](https://github.com/RangiLyu/nanodet)를 기반으로 합니다. `nanodet_src/`에는 이 프로젝트에서 사용하던 소스와 로컬 수정 사항을 포함했습니다.

원본 저장소의 기준 커밋: `be9b4a9001d7f9b6fc89c2df31ae8d428e35b4f0`

NanoDet의 원본 라이선스는 [nanodet_src/LICENSE](nanodet_src/LICENSE)이며, 기존 파일별 저작권 표시와 하위 구성 요소의 라이선스를 함께 유지합니다. 원본 사용법은 [nanodet_src/README.md](nanodet_src/README.md)를 참고하세요.
