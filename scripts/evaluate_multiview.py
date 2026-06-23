import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

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
    parser = argparse.ArgumentParser(description="Evaluate multiview AMR models.")
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--model", choices=MODEL_NAMES, default=None)
    parser.add_argument("--data-path", type=Path, default=Path("data/raw/RML2016.10a_dict.pkl"))
    parser.add_argument("--split-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--results-dir", type=Path, default=Path("results/multiview/eval"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--feature-dim", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--structure-alpha", type=float, default=None)
    parser.add_argument("--fft-shift", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--fft-transform", choices=FFT_TRANSFORMS, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(name)


def config_get(config: Dict[str, Any], name: str, default: Any) -> Any:
    value = config.get(name, default)
    return default if value is None else value


def write_rows(path: Path, fieldnames: List[str], rows: Iterable[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def accuracy_rows(
    stats: Dict[Tuple[Any, ...], Dict[str, int]],
    key_names: List[str],
    sort_key=None,
) -> List[dict]:
    rows = []
    for key, value in stats.items():
        key_tuple = key if isinstance(key, tuple) else (key,)
        total = value["total"]
        correct = value["correct"]
        row = {name: key_tuple[i] for i, name in enumerate(key_names)}
        row.update({"total": total, "correct": correct, "accuracy": correct / max(total, 1)})
        rows.append(row)
    return sorted(rows, key=sort_key)


def gate_mean_rows(
    stats: Dict[Tuple[Any, ...], Dict[str, float]],
    key_names: List[str],
    sort_key=None,
) -> List[dict]:
    rows = []
    for key, value in stats.items():
        key_tuple = key if isinstance(key, tuple) else (key,)
        total = int(value["total"])
        row = {name: key_tuple[i] for i, name in enumerate(key_names)}
        row.update(
            {
                "total": total,
                "mean_w_iq": value["w_iq"] / max(total, 1),
                "mean_w_ap": value["w_ap"] / max(total, 1),
                "mean_w_fft": value["w_fft"] / max(total, 1),
            }
        )
        rows.append(row)
    return sorted(rows, key=sort_key)


def increment_accuracy(stats: dict, key: Tuple[Any, ...], correct: bool) -> None:
    stats[key]["total"] += 1
    stats[key]["correct"] += int(correct)


def increment_gate(stats: dict, key: Tuple[Any, ...], weights: np.ndarray) -> None:
    stats[key]["total"] += 1
    stats[key]["w_iq"] += float(weights[0])
    stats[key]["w_ap"] += float(weights[1])
    stats[key]["w_fft"] += float(weights[2])


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
    ckpt_config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}

    model_name = args.model or checkpoint.get("model_name") or config_get(ckpt_config, "model", "ssg_gate")
    q_stats = checkpoint.get("q_stats")
    if model_name in ("ssg_gate", "ssg_gated_concat") and q_stats is None:
        raise RuntimeError(
            "This checkpoint does not contain q_stats. "
            "SSG models must be evaluated with train-set q statistics saved in the checkpoint."
        )
    feature_dim = int(args.feature_dim or config_get(ckpt_config, "feature_dim", 64))
    dropout = float(args.dropout if args.dropout is not None else config_get(ckpt_config, "dropout", 0.3))
    structure_alpha = float(
        args.structure_alpha
        if args.structure_alpha is not None
        else config_get(ckpt_config, "structure_alpha", 0.2)
    )
    fft_shift = bool(args.fft_shift if args.fft_shift is not None else config_get(ckpt_config, "fft_shift", True))
    fft_transform = args.fft_transform or config_get(ckpt_config, "fft_transform", "log1p")

    args.results_dir.mkdir(parents=True, exist_ok=True)

    dataset = RadioML2016Dataset(
        data_path=args.data_path,
        split_dir=args.split_dir,
        split=args.split,
        fft_shift=fft_shift,
        fft_transform=fft_transform,
        max_samples=args.max_samples,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    idx_to_class_raw = checkpoint.get("idx_to_class", dataset.idx_to_class)
    idx_to_class = {int(k): str(v) for k, v in idx_to_class_raw.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    class_order = {name: i for i, name in enumerate(class_names)}

    model = build_model(
        model_name,
        feature_dim=feature_dim,
        num_classes=len(class_names),
        dropout=dropout,
        fft_shift=fft_shift,
        structure_alpha=structure_alpha,
        q_stats=q_stats,
    ).to(device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    confusion = np.zeros((len(class_names), len(class_names)), dtype=np.int64)

    prediction_rows = []
    gate_rows = []
    acc_by_snr = defaultdict(lambda: {"total": 0, "correct": 0})
    acc_by_mod = defaultdict(lambda: {"total": 0, "correct": 0})
    acc_by_mod_snr = defaultdict(lambda: {"total": 0, "correct": 0})
    gate_by_snr = defaultdict(lambda: {"total": 0, "w_iq": 0.0, "w_ap": 0.0, "w_fft": 0.0})
    gate_by_mod = defaultdict(lambda: {"total": 0, "w_iq": 0.0, "w_ap": 0.0, "w_fft": 0.0})

    with torch.no_grad():
        for batch in loader:
            iq = batch["iq"].to(device, non_blocking=True)
            ap = batch["ap"].to(device, non_blocking=True)
            fft = batch["fft"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            output = model(iq, ap, fft, return_aux=True)
            logits = output["logits"] if isinstance(output, dict) else output
            weights = output.get("gate_weights") if isinstance(output, dict) else None

            loss = criterion(logits, labels)
            preds = logits.argmax(dim=1)
            correct_tensor = preds == labels

            labels_np = labels.cpu().numpy()
            preds_np = preds.cpu().numpy()
            sample_ids_np = batch["sample_id"].numpy()
            snrs_np = batch["snr"].numpy()
            correct_np = correct_tensor.cpu().numpy()
            weights_np = weights.cpu().numpy() if weights is not None else None

            total_loss += float(loss.item())
            total_correct += int(correct_tensor.sum().item())
            total_seen += int(labels.numel())

            for i in range(labels_np.shape[0]):
                true_label = int(labels_np[i])
                pred_label = int(preds_np[i])
                true_mod = idx_to_class[true_label]
                pred_mod = idx_to_class[pred_label]
                snr = int(snrs_np[i])
                sample_id = int(sample_ids_np[i])
                correct = bool(correct_np[i])
                confusion[true_label, pred_label] += 1

                prediction_rows.append(
                    {
                        "sample_id": sample_id,
                        "true_label": true_label,
                        "pred_label": pred_label,
                        "true_modulation": true_mod,
                        "pred_modulation": pred_mod,
                        "snr": snr,
                        "correct": int(correct),
                    }
                )

                increment_accuracy(acc_by_snr, (snr,), correct)
                increment_accuracy(acc_by_mod, (true_mod,), correct)
                increment_accuracy(acc_by_mod_snr, (true_mod, snr), correct)

                if weights_np is not None:
                    w = weights_np[i]
                    gate_rows.append(
                        {
                            "sample_id": sample_id,
                            "modulation": true_mod,
                            "snr": snr,
                            "w_iq": float(w[0]),
                            "w_ap": float(w[1]),
                            "w_fft": float(w[2]),
                            "correct": int(correct),
                        }
                    )
                    increment_gate(gate_by_snr, (snr,), w)
                    increment_gate(gate_by_mod, (true_mod,), w)

    metrics = {
        "model": model_name,
        "split": args.split,
        "checkpoint_path": args.checkpoint_path.as_posix(),
        "num_samples": total_seen,
        "loss": total_loss / max(total_seen, 1),
        "accuracy": total_correct / max(total_seen, 1),
        "correct": total_correct,
        "feature_dim": feature_dim,
        "dropout": dropout,
        "structure_alpha": structure_alpha,
        "fft_shift": fft_shift,
        "fft_transform": fft_transform,
        "q_stats_loaded": q_stats is not None,
        "gate_weights_saved": len(gate_rows) > 0,
        "class_names": class_names,
    }
    with (args.results_dir / "overall_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    write_rows(
        args.results_dir / "predictions.csv",
        [
            "sample_id",
            "true_label",
            "pred_label",
            "true_modulation",
            "pred_modulation",
            "snr",
            "correct",
        ],
        prediction_rows,
    )

    write_rows(
        args.results_dir / "accuracy_by_snr.csv",
        ["snr", "total", "correct", "accuracy"],
        accuracy_rows(acc_by_snr, ["snr"], sort_key=lambda r: int(r["snr"])),
    )
    write_rows(
        args.results_dir / "accuracy_by_modulation.csv",
        ["modulation", "total", "correct", "accuracy"],
        accuracy_rows(
            acc_by_mod, ["modulation"], sort_key=lambda r: class_order[str(r["modulation"])]
        ),
    )
    write_rows(
        args.results_dir / "accuracy_by_modulation_snr.csv",
        ["modulation", "snr", "total", "correct", "accuracy"],
        accuracy_rows(
            acc_by_mod_snr,
            ["modulation", "snr"],
            sort_key=lambda r: (class_order[str(r["modulation"])], int(r["snr"])),
        ),
    )

    confusion_rows = []
    for true_idx, true_name in enumerate(class_names):
        row = {"true_modulation": true_name}
        row.update({class_names[pred_idx]: int(confusion[true_idx, pred_idx]) for pred_idx in range(len(class_names))})
        confusion_rows.append(row)
    write_rows(args.results_dir / "confusion_matrix.csv", ["true_modulation"] + class_names, confusion_rows)

    if gate_rows:
        write_rows(
            args.results_dir / "gate_weights.csv",
            ["sample_id", "modulation", "snr", "w_iq", "w_ap", "w_fft", "correct"],
            gate_rows,
        )
        write_rows(
            args.results_dir / "gate_weights_by_snr.csv",
            ["snr", "total", "mean_w_iq", "mean_w_ap", "mean_w_fft"],
            gate_mean_rows(gate_by_snr, ["snr"], sort_key=lambda r: int(r["snr"])),
        )
        write_rows(
            args.results_dir / "gate_weights_by_modulation.csv",
            ["modulation", "total", "mean_w_iq", "mean_w_ap", "mean_w_fft"],
            gate_mean_rows(
                gate_by_mod,
                ["modulation"],
                sort_key=lambda r: class_order[str(r["modulation"])],
            ),
        )

    print(
        f"Evaluation finished: split={args.split} samples={total_seen} "
        f"accuracy={metrics['accuracy']:.6f} loss={metrics['loss']:.6f}"
    )
    print(f"Results dir: {args.results_dir}")


if __name__ == "__main__":
    main()
