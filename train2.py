#!/usr/bin/env python3
import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

# === Wandb ===
from pytorch_lightning.loggers import WandbLogger
import wandb

# 既存ルート（従来の単ビュー学習）
from datamodule.data_module import DataModule
from lightning import ModelModule

# 追加ルート（pre-contrastive：2ビュー＋TextGridで対照学習）
from datamodule.data_module_contrastive import ContrastiveDataModule
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

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ctc-weight", type=float, default=0.1)

    # ========= pre-contrastive =========
    parser.add_argument("--pre-contrastive", action="store_true",
                        help="Use contrastive route with (video1, video2, TextGrid).")
    parser.add_argument("--pair-train-file", type=str, default=None,
                        help="CSV for contrastive training pairs: dataset,rel_v1,rel_v2,textgrid,input_len,token")
    parser.add_argument("--pair-val-file", type=str, default=None)
    parser.add_argument("--pair-test-file", type=str, default=None)
    parser.add_argument("--fps", type=float, default=25.0,
                        help="FPS used to map TextGrid time (sec) to input frames.")
    parser.add_argument("--tg-tier", type=str, default="phones",
                        help="TextGrid tier name for phonemes.")
    parser.add_argument("--contrastive-weight", type=float, default=0.5,
                        help="Weight for contrastive loss in pre-contrastive route.")

    # ========= 既存の転移学習系 =========
    parser.add_argument("--pretrained_model_path", type=str, default=None)
    parser.add_argument("--transfer_frontend", action="store_true")
    parser.add_argument("--transfer_encoder", action="store_true")

    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--warmup_steps", type=int, default=1000)

    # バケツローダ用
    parser.add_argument("--max_frames", type=int, default=1000)

    # ========= W&B =========
    parser.add_argument("--wandb-project", type=str, default="auto_avsr_lipreader")
    parser.add_argument("--group-name", type=str, default=None, help="W&B group name")
    parser.add_argument("--wandb-offline", action="store_true", help="Use offline mode (no network)")
    parser.add_argument("--disable-wandb", action="store_true", help="Disable WandbLogger explicitly")

    args = parser.parse_args()
    return args


def get_trainer(args):
    os.makedirs(os.path.join(args.exp_dir, args.exp_name), exist_ok=True)

    # --- callbacks ---
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(args.exp_dir, args.exp_name),
            filename="{epoch:03d}-{loss:.3f}",
            monitor="loss",
            mode="min",
            save_top_k=3,
            save_last=True,
            every_n_epochs=10,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    # --- Wandb logger（“正常に記録される方”と同等の作り） ---
    logger = True  # True = Lightning がデフォルトの logger を作る（W&B無効時）
    if not args.disable_wandb:
        if args.wandb_offline:
            os.environ["WANDB_MODE"] = "offline"
        logger = WandbLogger(
            name=args.exp_name,
            project=args.wandb_project,
            group=args.group_name,
            save_dir=os.path.join(args.exp_dir, args.exp_name),
            log_model=True,
        )

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
        logger=logger,  # ★ ここが肝心
    )
    return trainer


def cli_main():
    args = parse_args()

    # ルート切替：pre-contrastive のときだけ対照学習
    if getattr(args, "pre_contrastive", False):
        modelmodule = ContrastiveModelModule(args)
        datamodule = ContrastiveDataModule(
            args=args,
            batch_size=args.batch_size,
            train_num_buckets=50,
            train_shuffle=True,
            num_workers=args.num_workers,
        )
    else:
        modelmodule = ModelModule(args)
        datamodule = DataModule(
            args=args,
            batch_size=args.batch_size,
            train_num_buckets=50,
            train_shuffle=True,
            num_workers=args.num_workers,
        )

    trainer = get_trainer(args)

    try:
        # 訓練
        trainer.fit(model=modelmodule, datamodule=datamodule, ckpt_path=args.ckpt_path)

        # テスト（必要なときだけ）
        if (args.test_file is not None) or (args.pair_test_file is not None):
            trainer.test(model=modelmodule, datamodule=datamodule)
    finally:
        # マルチGPUや例外時の取りこぼし防止
        try:
            wandb.finish()
        except Exception:
            pass


if __name__ == "__main__":
    cli_main()
