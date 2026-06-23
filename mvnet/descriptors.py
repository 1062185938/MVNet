import math
from typing import Dict

import torch


STRUCTURE_DESCRIPTOR_DIMS = {"iq": 2, "ap": 3, "fft": 3}


def _complex_from_iq(iq: torch.Tensor) -> torch.Tensor:
    if iq.ndim != 3 or iq.shape[1] != 2:
        raise ValueError(f"Expected IQ tensor shape [B, 2, N], got {tuple(iq.shape)}.")
    return torch.complex(iq[:, 0], iq[:, 1])


def phase_diff_from_iq(iq: torch.Tensor) -> torch.Tensor:
    signal = _complex_from_iq(iq)
    phase_diff = torch.zeros_like(iq[:, 0])
    phase_diff[:, 1:] = torch.angle(signal[:, 1:] * torch.conj(signal[:, :-1]))
    return phase_diff


def iq_descriptors(iq: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    power = iq[:, 0].pow(2) + iq[:, 1].pow(2)
    mean_power = power.mean(dim=1)
    papr = power.max(dim=1).values / (mean_power + eps)

    diff_i = iq[:, 0, 1:] - iq[:, 0, :-1]
    diff_q = iq[:, 1, 1:] - iq[:, 1, :-1]
    diff_power = diff_i.pow(2) + diff_q.pow(2)
    normalized_diff_energy = diff_power.mean(dim=1) / (mean_power + eps)

    return torch.stack([papr, normalized_diff_energy], dim=1)


def ap_descriptors(iq: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    amplitude = torch.sqrt(iq[:, 0].pow(2) + iq[:, 1].pow(2) + eps)
    amp_cv = amplitude.std(dim=1, unbiased=False) / (amplitude.mean(dim=1) + eps)

    phase_diff = phase_diff_from_iq(iq)
    phase_diff_std = phase_diff.std(dim=1, unbiased=False)
    phase_vector_mean_real = torch.cos(phase_diff).mean(dim=1)
    phase_vector_mean_imag = torch.sin(phase_diff).mean(dim=1)
    phase_coherence = torch.sqrt(
        phase_vector_mean_real.pow(2) + phase_vector_mean_imag.pow(2) + eps
    )

    return torch.stack([amp_cv, phase_diff_std, phase_coherence], dim=1)


def fft_descriptors(iq: torch.Tensor, fft_shift: bool = True, eps: float = 1e-8) -> torch.Tensor:
    signal = _complex_from_iq(iq)
    spectrum = torch.fft.fft(signal, dim=-1)
    if fft_shift:
        spectrum = torch.fft.fftshift(spectrum, dim=-1)

    power = spectrum.abs().pow(2)
    power_sum = power.sum(dim=1)
    prob = power / (power_sum[:, None] + eps)
    spectral_entropy = -(prob * (prob + eps).log()).sum(dim=1) / math.log(power.shape[1])

    spectral_flatness = torch.exp((power + eps).log().mean(dim=1)) / (power.mean(dim=1) + eps)
    peak_ratio = power.max(dim=1).values / (power_sum + eps)

    return torch.stack([spectral_entropy, spectral_flatness, peak_ratio], dim=1)


def compute_structure_descriptors(
    iq: torch.Tensor, fft_shift: bool = True, eps: float = 1e-8
) -> Dict[str, torch.Tensor]:
    return {
        "iq": iq_descriptors(iq, eps=eps),
        "ap": ap_descriptors(iq, eps=eps),
        "fft": fft_descriptors(iq, fft_shift=fft_shift, eps=eps),
    }


def normalize_structure_descriptors(
    descriptors: Dict[str, torch.Tensor],
    q_stats: Dict[str, Dict[str, torch.Tensor]],
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    normalized = {}
    for view, q_raw in descriptors.items():
        mean = torch.as_tensor(q_stats[view]["mean"], device=q_raw.device, dtype=q_raw.dtype)
        std = torch.as_tensor(q_stats[view]["std"], device=q_raw.device, dtype=q_raw.dtype)
        normalized[view] = (q_raw - mean) / (std + eps)
    return normalized
