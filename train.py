import os
import logging
import torch
import pytorch_lightning as pl
import io
import warnings

from contextlib import redirect_stdout
from pathlib import Path
from torch.utils.data import DataLoader
from tabulate import tabulate
from pytorch_lightning.callbacks import EarlyStopping

from nanodet.data.dataset import build_dataset
from nanodet.data.collate import naive_collate
from nanodet.evaluator import build_evaluator
from nanodet.trainer.task import TrainingTask
from nanodet.util import cfg,load_config,load_model_weight,NanoDetLightningLogger


class AlignedProgress(pl.Callback):
    columns = (
        ("mAP50", "AP_50", 100.0),
        ("mAP50-95", "mAP", 100.0),
        ("precision", "precision", 100.0),
        ("recall", "recall", 100.0),
        ("count", "count", 1.0),
    )

    def __init__(self, every=100):
        super().__init__()
        self.every = every

    def on_train_epoch_start(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return

        header = (
            f"\n{'PHASE':<5} {'EPOCH':>7} {'BATCH':>11} "
            f"{'PROGRESS':>10} {'LOSS':>10}"
        )

        for name, _, _ in self.columns:
            header += f" {name:>10}"

        print(header, flush=True)

    def _show(
        self, trainer, phase, done, total,
        loss=None, metrics=None
    ):
        if not trainer.is_global_zero or total == 0:
            return

        # 지정한 간격 또는 마지막 배치에서 출력
        if done % self.every != 0 and done != total:
            return

        epoch_text = f"{trainer.current_epoch + 1}/{trainer.max_epochs}"
        batch_text = f"{done}/{total}"
        progress = 100.0 * done / total

        # TRAIN loss는 Tensor, 평균 VAL loss는 float
        if torch.is_tensor(loss):
            loss = loss.detach().item()

        loss_text = "--" if loss is None else f"{float(loss):.4f}"

        row = (
            f"{phase:<5} {epoch_text:>7} {batch_text:>11} "
            f"{progress:9.1f}% {loss_text:>10}"
        )

        for _, key, scale in self.columns:
            if metrics is None:
                value_text = "--"
            else:
                value_text = f"{float(metrics[key]) * scale:.2f}%"

            row += f" {value_text:>10}"

        print(row, flush=True)

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx
    ):
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs

        self._show(
            trainer,
            "TRAIN",
            batch_idx + 1,
            int(trainer.num_training_batches),
            loss=loss,
        )

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx,
        dataloader_idx=0
    ):
        if trainer.sanity_checking:
            return

        done = batch_idx + 1
        total = int(trainer.num_val_batches[dataloader_idx])

        # 마지막 VAL 행은 평가 지표 계산 후 출력
        if done < total:
            self._show(trainer, "VAL", done, total)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return

        metrics = getattr(pl_module, "latest_val_metrics", None)

        if metrics is None or pl_module.val_sample_count == 0:
            return

        mean_loss = (
            pl_module.val_loss_sum / pl_module.val_sample_count
        )
        total = int(trainer.num_val_batches[0])

        self._show(
            trainer,
            "VAL",
            total,
            total,
            loss=mean_loss,
            metrics=metrics,
        )


class HideBatchMessages(logging.Filter):
    def filter(self, record):
        if record.levelno >= logging.WARNING:
            return True

        return not record.getMessage().startswith(("Train|", "Val|"))


def main():
    # Lightning의 일반 안내 메시지 숨기기
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    logging.getLogger("pytorch_lightning.callbacks.early_stopping").setLevel(logging.INFO)
    logging.getLogger("lightning_fabric").setLevel(logging.WARNING)

    # 현재 num_workers 설정에 대한 성능 권장 메시지 숨기기
    warnings.filterwarnings(
        "ignore",
        message=r"The dataloader, .*does not have many workers.*",
        category=UserWarning,
        module=r"pytorch_lightning\.trainer\.connectors\.data_connector",
    )

    # meshgrid의 향후 API 변경 안내 숨기기
    warnings.filterwarnings(
        "ignore",
        message=r"torch\.meshgrid: in an upcoming release,.*",
        category=UserWarning,
        module=r"torch\.functional",
    )
    
    # 프로젝트 디렉토리 설정
    project_dir = Path(__file__).resolve().parent
    os.chdir(project_dir)

    # 설정 파일 로드
    config_path = project_dir / "configs" / "nanodet-plus-m_416_can.yml"
    load_config(cfg, str(config_path))

    # 무작위 시드 고정
    pl.seed_everything(42, workers=True)

    # 학습 & 검증 데이터셋 빌드
    with redirect_stdout(io.StringIO()):
        train_dataset = build_dataset(cfg.data.train, mode="train")
        val_dataset = build_dataset(cfg.data.val, mode="test")


    # 데이터셋 예외 처리: 데이터셋이 비어 있는 경우 오류 발생
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("데이터셋이 비어 있습니다. 이미지와 라벨 경로를 확인하세요.")


    # 데이터셋 샘플 확인
    sample = val_dataset[0]


    # 데이터 로더 생성
    batch_size = cfg.device.batchsize_per_gpu

    if len(train_dataset) < batch_size:
        raise ValueError("학습 이미지 수보다 배치 크기가 큽니다.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg.device.workers_per_gpu,
        collate_fn=naive_collate,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.device.workers_per_gpu,
        collate_fn=naive_collate,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


    # 로거 및 평가자 설정
    logger = NanoDetLightningLogger(cfg.save_dir)
    logger.logger.propagate = False

    # 콘솔에는 WARNING 이상만 표시
    for handler in logger.logger.handlers:
        if (
            isinstance(handler, logging.StreamHandler)
            and not isinstance(handler, logging.FileHandler)
        ):
            handler.setLevel(logging.WARNING)

    logger.dump_cfg(cfg)

    evaluator = build_evaluator(cfg.evaluator, val_dataset)

    # 모델 초기화 중의 일반 출력 숨기기
    with redirect_stdout(io.StringIO()):
        task = TrainingTask(cfg, evaluator)


    # 사전 학습 가중치 로드
    checkpoint_path = project_dir / cfg.schedule.load_model

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"체크포인트가 없습니다: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    load_model_weight(task.model, checkpoint, logger)


    # 학습 GPU 확인
    if cfg.device.gpu_ids == -1:
        accelerator = "cpu"
        devices = 1
        device_name = "CPU"
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("GPU 학습으로 설정되어 있지만 CUDA를 사용할 수 없습니다.")

        accelerator = "gpu"
        devices = cfg.device.gpu_ids
        device_name = torch.cuda.get_device_name(devices[0])


    # Early stopping 콜백 설정
    early_stopping = EarlyStopping(
        monitor="val_mAP",
        mode="max",
        patience=10,
        min_delta=0.001,
        check_on_train_epoch_end=False,
        strict=True,
        verbose=True,
    )

    # 학습 시작
    trainer = pl.Trainer(
        default_root_dir=cfg.save_dir,
        accelerator=accelerator,
        devices=devices,
        precision=cfg.device.precision,
        max_epochs=cfg.schedule.total_epochs,
        check_val_every_n_epoch=cfg.schedule.val_intervals,
        log_every_n_steps=cfg.log.interval,
        gradient_clip_val=cfg.grad_clip,
        num_sanity_val_steps=0,
        logger=logger,
        enable_model_summary=False,
        callbacks=[AlignedProgress(every=5), early_stopping],
        enable_progress_bar=False,
        limit_train_batches=20,
        limit_val_batches=5,
    )

    # 학습 시작 정보
    input_width, input_height = cfg.data.train.input_size

    # 학습 시작 정보 출력
    summary_rows = [
        ["모델", cfg.model.arch.name],
        ["학습 장치", device_name],
        ["PyTorch", torch.__version__],
        ["클래스", ", ".join(cfg.class_names)],
        ["학습 이미지", f"{len(train_dataset):,}장"],
        ["검증 이미지", f"{len(val_dataset):,}장"],
        ["입력 크기", f"{input_width} x {input_height}"],
        ["배치 크기", f"{batch_size}장"],
        ["학습 횟수", f"{cfg.schedule.total_epochs} epoch"],
        ["가중치 로드", "완료"],
    ]

    print("\n[학습 시작 정보]", flush=True)
    print( tabulate( summary_rows, headers=["항목", "값"], tablefmt="rounded_outline", colalign=("left", "left"), disable_numparse=True), flush=True)

    trainer.fit(
        model=task,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

if __name__ == "__main__":
    main()