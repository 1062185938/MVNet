import argparse
import json
import pickle
from pathlib import Path

import numpy as np


def load_radioml_dict(path: Path):
    with path.open("rb") as f:
        return pickle.load(f, encoding="latin1")


def build_group_table(raw_data):
    mods = sorted({key[0] for key in raw_data.keys()})
    snrs = sorted({int(key[1]) for key in raw_data.keys()})

    groups = []
    offset = 0
    for mod in mods:
        for snr in snrs:
            key = (mod, snr)
            if key not in raw_data:
                continue
            num_samples = int(raw_data[key].shape[0])
            groups.append(
                {
                    "modulation": mod,
                    "snr": snr,
                    "start_index": offset,
                    "end_index": offset + num_samples,
                    "num_samples": num_samples,
                }
            )
            offset += num_samples

    return mods, snrs, groups, offset


def split_group_indices(start, num_samples, rng, train_ratio, val_ratio):
    indices = np.arange(start, start + num_samples, dtype=np.int64)
    rng.shuffle(indices)

    train_count = int(num_samples * train_ratio)
    val_count = int(num_samples * val_ratio)

    train_indices = indices[:train_count]
    val_indices = indices[train_count : train_count + val_count]
    test_indices = indices[train_count + val_count :]

    return train_indices, val_indices, test_indices


def create_splits(data_path: Path, output_dir: Path, seed: int):
    raw_data = load_radioml_dict(data_path)
    mods, snrs, groups, total_samples = build_group_table(raw_data)

    rng = np.random.default_rng(seed)
    train_parts = []
    val_parts = []
    test_parts = []
    split_counts_by_group = []

    for group in groups:
        train_idx, val_idx, test_idx = split_group_indices(
            group["start_index"],
            group["num_samples"],
            rng,
            train_ratio=0.6,
            val_ratio=0.2,
        )
        train_parts.append(train_idx)
        val_parts.append(val_idx)
        test_parts.append(test_idx)
        split_counts_by_group.append(
            {
                "modulation": group["modulation"],
                "snr": group["snr"],
                "total": group["num_samples"],
                "train": int(train_idx.shape[0]),
                "val": int(val_idx.shape[0]),
                "test": int(test_idx.shape[0]),
            }
        )

    train_indices = np.concatenate(train_parts).astype(np.int64)
    val_indices = np.concatenate(val_parts).astype(np.int64)
    test_indices = np.concatenate(test_parts).astype(np.int64)

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "train_indices.npy", train_indices)
    np.save(output_dir / "val_indices.npy", val_indices)
    np.save(output_dir / "test_indices.npy", test_indices)

    meta = {
        "dataset": "RadioML2016.10A",
        "data_path": str(data_path.as_posix()),
        "split_dir": str(output_dir.as_posix()),
        "seed": seed,
        "ratios": {"train": 0.6, "val": 0.2, "test": 0.2},
        "index_policy": {
            "description": "Global sample_id is assigned by sorted modulation, then sorted SNR, then local sample index within each RadioML array.",
            "modulation_order": mods,
            "snr_order": snrs,
        },
        "num_classes": len(mods),
        "class_to_idx": {mod: idx for idx, mod in enumerate(mods)},
        "idx_to_class": {str(idx): mod for idx, mod in enumerate(mods)},
        "total_samples": int(total_samples),
        "split_counts": {
            "train": int(train_indices.shape[0]),
            "val": int(val_indices.shape[0]),
            "test": int(test_indices.shape[0]),
        },
        "groups": groups,
        "split_counts_by_group": split_counts_by_group,
        "files": {
            "train_indices": "train_indices.npy",
            "val_indices": "val_indices.npy",
            "test_indices": "test_indices.npy",
            "split_meta": "split_meta.json",
        },
    }

    with (output_dir / "split_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return meta


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create reproducible stratified index splits for RadioML2016.10A."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/raw/RML2016.10a_dict.pkl"),
        help="Path to the original RadioML pickle file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/splits"),
        help="Directory for train/val/test index files and split metadata.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main():
    args = parse_args()
    meta = create_splits(args.data_path, args.output_dir, args.seed)
    print("Created RadioML split index files.")
    print(f"Total samples: {meta['total_samples']}")
    print(f"Train: {meta['split_counts']['train']}")
    print(f"Val: {meta['split_counts']['val']}")
    print(f"Test: {meta['split_counts']['test']}")
    print(f"Output dir: {args.output_dir}")


if __name__ == "__main__":
    main()
