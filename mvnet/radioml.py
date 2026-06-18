import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


FFT_TRANSFORMS = ("log1p", "standardize", "log1p_standardize", "none")


def load_radioml_dict(path: Path) -> Dict[Tuple[str, int], np.ndarray]:
    with Path(path).open("rb") as f:
        return pickle.load(f, encoding="latin1")


def load_split_meta(split_dir: Path) -> dict:
    meta_path = Path(split_dir) / "split_meta.json"
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_iq_view(sample: np.ndarray) -> np.ndarray:
    sample = np.asarray(sample, dtype=np.float32)
    if sample.ndim != 2 or sample.shape[0] != 2:
        raise ValueError(f"Expected IQ sample shape [2, N], got {sample.shape}.")
    return np.ascontiguousarray(sample)


def build_ap_view(iq: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    i = iq[0]
    q = iq[1]
    amplitude = np.sqrt(i * i + q * q + eps).astype(np.float32)

    signal = i.astype(np.float32) + 1j * q.astype(np.float32)
    phase_diff = np.zeros_like(i, dtype=np.float32)
    phase_diff[1:] = np.angle(signal[1:] * np.conj(signal[:-1])).astype(np.float32)

    return np.ascontiguousarray(np.stack([amplitude, phase_diff], axis=0))


def build_fft_view(
    iq: np.ndarray,
    fft_shift: bool = True,
    transform: str = "log1p",
    eps: float = 1e-8,
) -> np.ndarray:
    if transform not in FFT_TRANSFORMS:
        raise ValueError(f"Unknown FFT transform '{transform}'. Use one of {FFT_TRANSFORMS}.")

    signal = iq[0].astype(np.float32) + 1j * iq[1].astype(np.float32)
    fft_mag = np.abs(np.fft.fft(signal)).astype(np.float32)
    if fft_shift:
        fft_mag = np.fft.fftshift(fft_mag).astype(np.float32)

    if transform in ("log1p", "log1p_standardize"):
        fft_mag = np.log1p(fft_mag).astype(np.float32)

    if transform in ("standardize", "log1p_standardize"):
        fft_mag = ((fft_mag - fft_mag.mean()) / (fft_mag.std() + eps)).astype(np.float32)

    return np.ascontiguousarray(fft_mag[None, :])


class RadioML2016Dataset(Dataset):
    """RadioML2016.10A split dataset with IQ/AP/FFT views."""

    SPLIT_FILES = {
        "train": "train_indices.npy",
        "val": "val_indices.npy",
        "test": "test_indices.npy",
    }

    def __init__(
        self,
        data_path: Path,
        split_dir: Path,
        split: str,
        fft_shift: bool = True,
        fft_transform: str = "log1p",
        max_samples: Optional[int] = None,
        raw_data: Optional[Dict[Tuple[str, int], np.ndarray]] = None,
        meta: Optional[dict] = None,
    ) -> None:
        if split not in self.SPLIT_FILES:
            raise ValueError(f"Unknown split '{split}'. Use one of {sorted(self.SPLIT_FILES)}.")

        self.data_path = Path(data_path)
        self.split_dir = Path(split_dir)
        self.split = split
        self.fft_shift = fft_shift
        self.fft_transform = fft_transform

        self.meta = meta if meta is not None else load_split_meta(self.split_dir)
        self.raw_data = raw_data if raw_data is not None else load_radioml_dict(self.data_path)

        indices_path = self.split_dir / self.SPLIT_FILES[split]
        self.indices = np.load(indices_path).astype(np.int64)
        if max_samples is not None:
            self.indices = self.indices[: int(max_samples)]

        self.groups = self.meta["groups"]
        self.group_starts = np.asarray([g["start_index"] for g in self.groups], dtype=np.int64)
        self.group_ends = np.asarray([g["end_index"] for g in self.groups], dtype=np.int64)
        self.group_keys = [(g["modulation"], int(g["snr"])) for g in self.groups]
        self.group_snrs = np.asarray([int(g["snr"]) for g in self.groups], dtype=np.int64)

        self.class_to_idx = {str(k): int(v) for k, v in self.meta["class_to_idx"].items()}
        self.idx_to_class = {int(k): str(v) for k, v in self.meta["idx_to_class"].items()}
        self.class_names = [self.idx_to_class[i] for i in range(len(self.idx_to_class))]
        self.group_labels = np.asarray(
            [self.class_to_idx[g["modulation"]] for g in self.groups], dtype=np.int64
        )

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def locate_sample(self, sample_id: int) -> Tuple[int, int]:
        group_index = int(np.searchsorted(self.group_ends, sample_id, side="right"))
        if group_index < 0 or group_index >= len(self.groups):
            raise IndexError(f"Sample id {sample_id} is outside the split metadata range.")
        local_index = int(sample_id - self.group_starts[group_index])
        return group_index, local_index

    def __getitem__(self, index: int) -> dict:
        sample_id = int(self.indices[index])
        group_index, local_index = self.locate_sample(sample_id)
        key = self.group_keys[group_index]

        iq = build_iq_view(self.raw_data[key][local_index])
        ap = build_ap_view(iq)
        fft = build_fft_view(iq, fft_shift=self.fft_shift, transform=self.fft_transform)

        return {
            "iq": torch.from_numpy(iq),
            "ap": torch.from_numpy(ap),
            "fft": torch.from_numpy(fft),
            "label": torch.tensor(int(self.group_labels[group_index]), dtype=torch.long),
            "sample_id": torch.tensor(sample_id, dtype=torch.long),
            "snr": torch.tensor(int(self.group_snrs[group_index]), dtype=torch.long),
        }
