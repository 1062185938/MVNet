# Work Log

## 2026-06-18

- Added a reusable `mvnet` package for RadioML2016.10A multiview AMR experiments.
- Implemented IQ/AP/FFT view construction. AP uses amplitude plus padded phase difference; FFT supports `fftshift` and configurable magnitude transforms.
- Implemented batch signal-structure descriptors: IQ PAPR and normalized differential energy, AP amplitude CV/phase-diff std/phase coherence, and FFT spectral entropy/flatness/peak ratio.
- Implemented six model variants: `iq_cnn`, `ap_cnn`, `fft_cnn`, `concat`, `vanilla_gate`, and `ssg_gate`.
- Added `scripts/train_multiview.py` with reproducible seeding, config saving, CSV training log, and best-checkpoint saving.
- Added `scripts/evaluate_multiview.py` with overall metrics, predictions, SNR/modulation breakdowns, confusion matrix, and gate-weight exports for gate models.
- Verified with CPU smoke tests:
  - `ssg_gate` one-epoch training on 512 train / 256 val samples completed and saved `checkpoints/smoke_ssg_gate_best.pt`.
  - Evaluation on 256 validation samples completed and generated all requested metric/gate CSV files.
  - Forward-pass shape check passed for all six model variants.
