# 工作日志

## 2026-06-18

- 新增 `mvnet` 包，用于 RadioML2016.10A 自动调制识别的轻量多视图实验。
- 实现 IQ/AP/FFT 三视图构造：
  - IQ view 使用原始 `[I, Q]`，shape 为 `[2, 128]`。
  - AP view 使用 amplitude 和 padded phase difference，shape 为 `[2, 128]`。
  - FFT view 使用 `abs(fft(s))`，支持通过参数控制是否 `fftshift`，并默认使用 `log1p` 缩放，shape 为 `[1, 128]`。
- 实现 batch 级 signal-structure descriptors：
  - `q_iq`：PAPR、normalized differential energy。
  - `q_ap`：amp_cv、phase_diff_std、phase_coherence。
  - `q_fft`：spectral_entropy、spectral_flatness、peak_ratio。
- 实现六种模型：
  - `iq_cnn`
  - `ap_cnn`
  - `fft_cnn`
  - `concat`
  - `vanilla_gate`
  - `ssg_gate`
- 新增训练脚本 `scripts/train_multiview.py`，支持随机种子、自动设备选择、训练日志、配置保存和 best checkpoint 保存。
- 新增评估脚本 `scripts/evaluate_multiview.py`，支持整体指标、逐样本预测、按 SNR/调制方式统计、混淆矩阵，以及 gate 权重分析。
- 在 CPU 环境下完成小样本验证：
  - 使用 `ssg_gate` 在 512 个训练样本和 256 个验证样本上完成 1 个 epoch 训练。
  - 成功保存 `checkpoints/smoke_ssg_gate_best.pt`。
  - 成功生成训练日志、评估指标、预测结果、混淆矩阵和 gate 权重统计。
  - 六个模型分支均完成前向形状检查，输出 shape 均为 `[B, 11]`。

## 2026-06-18 追加

- 为 `SignalStructureGuidedGateCNN` 新增 `structure_alpha` 参数，默认值为 `0.2`。
- 将 SSG gate 分数从 `feat_score + struct_score` 调整为 `feat_score + structure_alpha * struct_score`。
- 对 `struct_score_iq`、`struct_score_ap`、`struct_score_fft` 的最后一层 `Linear` 做 zero initialization，使 `ssg_gate` 初始状态更接近 `vanilla_gate`。
- 新增模型 `ssg_gated_concat`：仍然使用 SSG gate 生成 `w_iq`、`w_ap`、`w_fft`，但融合方式改为 `concat(w_iq*z_iq, w_ap*z_ap, w_fft*z_fft)`，分类头输入维度为 `3 * feature_dim`。
- 更新 `MODEL_NAMES` 和 `build_model()`，支持 `ssg_gated_concat`，并支持传入 `structure_alpha`。
- 更新训练和评估脚本，支持命令行参数 `--structure-alpha`。
- 完成验证：
  - `python -m compileall mvnet scripts` 通过。
  - `ssg_gated_concat` 前向检查通过，输出 logits shape 为 `[B, 11]`，gate weights shape 为 `[B, 3]`，融合特征 shape 为 `[B, 192]`。
  - 确认结构分支最后一层权重和 bias 初始值为 0。
  - 使用 CPU 极小样本完成 `ssg_gated_concat` 训练入口冒烟测试。
