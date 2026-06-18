import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mvnet import MODEL_NAMES, RadioML2016Dataset, build_model
from mvnet.radioml import FFT_TRANSFORMS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train lightweight multiview AMR models.")
    parser.add_argument("--data-path", type=Path, default=Path("data/raw/RML2016.10a_dict.pkl"))
    parser.add_argument("--split-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--model", choices=MODEL_NAMES, default="ssg_gate")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--feature-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--results-dir", type=Path, default=Path("results/multiview/ssg_gate"))
    parser.add_argument("--checkpoint-path", type=Path, default=Path("checkpoints/ssg_gate_best.pt"))
    parser.add_argument("--fft-shift", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fft-transform", choices=FFT_TRANSFORMS, default="log1p")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(name)


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def make_loader(
    dataset: RadioML2016Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    device: torch.device,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=generator,
    )


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer = None,
) -> tuple:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for batch in loader:
        batch = move_batch(batch, device)
        labels = batch["label"]

        with torch.set_grad_enabled(training):
            logits = model(batch["iq"], batch["ap"], batch["fft"])
            loss = criterion(logits, labels)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        batch_size = int(labels.numel())
        total_loss += float(loss.item()) * batch_size
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_seen += batch_size

    return total_loss / max(total_seen, 1), total_correct / max(total_seen, 1)


def save_config(args: argparse.Namespace, device: torch.device, results_dir: Path) -> Dict[str, Any]:
    config = json_ready(vars(args))
    config["resolved_device"] = str(device)
    config["torch_version"] = torch.__version__
    config_path = results_dir / "config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    return config


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = select_device(args.device)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    config = save_config(args, device, args.results_dir)

    train_dataset = RadioML2016Dataset(
        data_path=args.data_path,
        split_dir=args.split_dir,
        split="train",
        fft_shift=args.fft_shift,
        fft_transform=args.fft_transform,
        max_samples=args.max_train_samples,
    )
    val_dataset = RadioML2016Dataset(
        data_path=args.data_path,
        split_dir=args.split_dir,
        split="val",
        fft_shift=args.fft_shift,
        fft_transform=args.fft_transform,
        max_samples=args.max_val_samples,
        raw_data=train_dataset.raw_data,
        meta=train_dataset.meta,
    )

    train_loader = make_loader(
        train_dataset, args.batch_size, True, args.num_workers, args.seed, device
    )
    val_loader = make_loader(
        val_dataset, args.batch_size, False, args.num_workers, args.seed + 1, device
    )

    model = build_model(
        args.model,
        feature_dim=args.feature_dim,
        num_classes=len(train_dataset.class_names),
        dropout=args.dropout,
        fft_shift=args.fft_shift,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    log_path = args.results_dir / "train_log.csv"
    best_val_acc = -1.0
    best_epoch = 0

    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"]
        )
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            start = time.time()
            train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
            val_loss, val_acc = run_epoch(model, val_loader, criterion, device)

            row = {
                "epoch": epoch,
                "train_loss": f"{train_loss:.6f}",
                "train_acc": f"{train_acc:.6f}",
                "val_loss": f"{val_loss:.6f}",
                "val_acc": f"{val_acc:.6f}",
            }
            writer.writerow(row)
            f.flush()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                torch.save(
                    {
                        "epoch": epoch,
                        "model_name": args.model,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_val_acc": best_val_acc,
                        "config": config,
                        "class_to_idx": train_dataset.class_to_idx,
                        "idx_to_class": train_dataset.idx_to_class,
                    },
                    args.checkpoint_path,
                )

            elapsed = time.time() - start
            print(
                f"epoch {epoch:03d}/{args.epochs:03d} "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
                f"best={best_val_acc:.4f}@{best_epoch} time={elapsed:.1f}s"
            )

    print(f"Training finished. Best val_acc={best_val_acc:.6f} at epoch {best_epoch}.")
    print(f"Log: {log_path}")
    print(f"Checkpoint: {args.checkpoint_path}")


if __name__ == "__main__":
    main()
