#!/usr/bin/env python3
import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

# 既存ルート（従来の単ビュー学習）
from datamodule.data_module import DataModule
from lightning import ModelModule

# 追加ルート（pre-contrastive：2ビュー＋TextGridで対照学習）
from datamodule.data_module import ContrastiveDataModule
from lightning import ContrastiveModelModule


def parse_args():
    parser = argparse.ArgumentParser()

    # ========= 既存オプション =========
    parser.add_argument("--exp-dir", type=str, default="./exp")
    parser.add_argument("--exp-name", type=str, default="run")
    parser.add_argument("--modality", type=str, default="video", choices=["video", "audio"])
    parser.add_argument("--root-dir", type=str, required=True)
    parser.add_argument("--train-file", type=str, default=None)
    parser.add_argument("--val-file", type=str, default=None)
    parser.add_argument("--test-file", type=str, default=None)

    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)

    parser.add_argument("--gpus", type=int, default=1)            # PL <=1.x 互換
    parser.add_argument("--num-nodes", type=int, default=1)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--ckpt-path", type=str, default=None)

    # 学習率/重み減衰（既存ModelModuleで参照）
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    # CTC:Attention 比率（ESPnet E2E 内で使用）
    parser.add_argument("--ctc-weight", type=float, default=0.1)

    # ========= 追加: pre-contrastive 用フラグ =========
    parser.add_argument(
        "--pre-contrastive",
        action="store_true",
        help="Use contrastive route with (video1, video2, TextGrid)."
    )

    # TextGrid 関連（必要に応じて Model/DM 側が参照）
    parser.add_argument("--fps", type=float, default=25.0, help="FPS used to map TextGrid time (sec) to input frames.")
    parser.add_argument("--tg-tier", type=str, default="phones", help="TextGrid tier name for phonemes.")
    parser.add_argument("--contrastive-weight", type=float, default=0.5, help="Weight for contrastive loss in pre-contrastive route.")

    # ========= 互換用: datamodule 側が参照する pair_* を定義だけしておく（既定 None）=========
    parser.add_argument("--pair-train-file", type=str, default=None,
                        help="(optional) CSV for contrastive training pairs")
    parser.add_argument("--pair-val-file", type=str, default=None,
                        help="(optional) CSV for contrastive val pairs")
    parser.add_argument("--pair-test-file", type=str, default=None,
                        help="(optional) CSV for contrastive test pairs")

    # ========= 既存の転移学習系（必要に応じてModelModule側で参照） =========
    parser.add_argument("--pretrained_model_path", type=str, default=None)
    parser.add_argument("--transfer_frontend", action="store_true")
    parser.add_argument("--transfer_encoder", action="store_true")

    # （必要なら）ウォームアップ・スケジューラ用
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--warmup_steps", type=int, default=1000)

    # バケツローダ用
    parser.add_argument("--max_frames", type=int, default=1000)

    args = parser.parse_args()
    return args


def get_trainer(args):
    os.makedirs(os.path.join(args.exp_dir, args.exp_name), exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(args.exp_dir, args.exp_name),
            filename="{epoch:03d}-{loss_val:.3f}",
            monitor="loss_val",   # train.py と同様に val loss を監視
            mode="min",
            save_top_k=3,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    # ★ 追加: W&B ロガー（最小追加）
    wandb_logger = WandbLogger(
        name=args.exp_name,
        project="auto_avsr_lipreader",  # 必要なら変更してください
    )
    # run の config に引数を入れておくと便利（任意）
    try:
        wandb_logger.experiment.config.update(vars(args), allow_val_change=True)
    except Exception:
        pass

    # 既存の実装に合わせ、devices 指定は gpus 値をそのまま利用
    accelerator = "gpu" if (args.gpus and args.gpus > 0) else "cpu"
    trainer = pl.Trainer(
        default_root_dir=os.path.join(args.exp_dir, args.exp_name),
        max_epochs=args.max_epochs,
        accelerator=accelerator,
        devices=args.gpus if accelerator == "gpu" else None,
        num_nodes=args.num_nodes,
        callbacks=callbacks,
        gradient_clip_val=1.0,
        log_every_n_steps=50,
        logger=wandb_logger,  # ★ ここが肝
    )
    return trainer


def cli_main():
    args = parse_args()

    # ========= pre-contrastive のときだけルートを切替 =========
    if getattr(args, "pre_contrastive", False):
        # 対照学習（2ビュー＋TextGrid）ルート
        modelmodule = ContrastiveModelModule(args)
        datamodule = ContrastiveDataModule(
            args=args,
            batch_size=args.batch_size,
            train_num_buckets=50,
            train_shuffle=True,
            num_workers=args.num_workers,
        )
    else:
        # 既存の単ビュー学習ルート
        modelmodule = ModelModule(args)
        datamodule = DataModule(
            args=args,
            batch_size=args.batch_size,
            train_num_buckets=50,
            train_shuffle=True,
            num_workers=args.num_workers,
        )

    trainer = get_trainer(args)

    # 訓練
    trainer.fit(model=modelmodule, datamodule=datamodule, ckpt_path=args.ckpt_path)

    # 検証 / テスト（必要に応じて）
    # if args.val_file is not None or args.pair_val_file is not None:
    #     trainer.validate(model=modelmodule, datamodule=datamodule)
    if (args.test_file is not None) or (args.pair_test_file is not None):
        trainer.test(model=modelmodule, datamodule=datamodule)


if __name__ == "__main__":
    cli_main()
