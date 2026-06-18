# MVNet RadioML2016.10A 多视图自动调制识别

本项目实现了一个用于 RadioML2016.10A 的轻量多视图自适应融合模型。每个 IQ 样本会被构造成三个视图：IQ、AP 和 FFT，并支持多个 baseline / ablation，便于后续实验对比。

## 数据路径

默认原始数据路径：

```powershell
data/raw/RML2016.10a_dict.pkl
```

默认划分文件目录：

```powershell
data/splits
```

`data/splits` 下需要包含：

```text
train_indices.npy
val_indices.npy
test_indices.npy
split_meta.json
```

如果需要重新生成划分，可以运行：

```powershell
python scripts/create_radioml_splits.py `
  --data-path data/raw/RML2016.10a_dict.pkl `
  --output-dir data/splits `
  --seed 42
```

## 支持的模型

训练脚本通过 `--model` 选择模型：

```text
iq_cnn        只使用 IQ view
ap_cnn        只使用 AP view
fft_cnn       只使用 FFT view
concat        三视图 concat 融合
vanilla_gate  只使用深度特征生成 gate 权重
ssg_gate      使用深度特征 + signal-structure descriptors 生成 gate 权重
ssg_gated_concat  使用 SSG gate 加权后再 concat 融合
```

### `--model` 选项说明

| 选项 | 使用的输入视图 | 融合方式 | 简单说明 |
| --- | --- | --- | --- |
| `iq_cnn` | IQ | 无融合 | 只使用原始 `[I, Q]` 信号训练一个 1D-CNN，是最基础的时域 IQ baseline。 |
| `ap_cnn` | AP | 无融合 | 只使用 amplitude 和 phase_diff，其中 phase_diff 是相邻复采样点的相位差，适合观察幅度变化和相位跳变信息。 |
| `fft_cnn` | FFT | 无融合 | 只使用频域幅度谱 `abs(fft(s))`，适合观察频谱分布、带宽和频域峰值结构。 |
| `concat` | IQ + AP + FFT | 特征拼接 | 三个分支分别提取 64 维特征，然后拼接成 192 维特征再分类；这是普通多视图融合 baseline。 |
| `vanilla_gate` | IQ + AP + FFT | 自适应加权融合 | 三个分支分别输出 64 维特征，模型只根据深度特征生成 `w_iq`、`w_ap`、`w_fft`，再加权融合。 |
| `ssg_gate` | IQ + AP + FFT + 结构指标 | 结构引导自适应加权融合 | 完整方法。gate 同时使用深度特征和 signal-structure descriptors，让模型学习在不同信号结构下更信任哪个视图。 |
| `ssg_gated_concat` | IQ + AP + FFT + 结构指标 | 结构引导加权后拼接 | 使用和 `ssg_gate` 相同的 SSG gate 生成权重，但不做 weighted-sum，而是拼接 `w_iq*z_iq`、`w_ap*z_ap`、`w_fft*z_fft` 后再分类。 |

单视图模型 `iq_cnn`、`ap_cnn`、`fft_cnn` 主要用于回答“某一个视图单独有多强”。`concat` 用于回答“普通三视图拼接是否有效”。`vanilla_gate` 用于验证“自适应 gate 是否比简单 concat 更好”。`ssg_gate` 用于验证结构指标是否能进一步帮助 gate 做出更合理的视图权重分配。`ssg_gated_concat` 则用于验证“保留三个视图各自的加权特征再分类”是否比直接加权求和更有效。

## 训练脚本

脚本路径：

```powershell
scripts/train_multiview.py
```

推荐的完整方法训练命令：

```powershell
python scripts/train_multiview.py `
  --data-path data/raw/RML2016.10a_dict.pkl `
  --split-dir data/splits `
  --model ssg_gate `
  --epochs 50 `
  --batch-size 256 `
  --lr 1e-3 `
  --weight-decay 1e-4 `
  --dropout 0.3 `
  --feature-dim 64 `
  --structure-alpha 0.2 `
  --seed 42 `
  --device auto `
  --results-dir results/multiview/ssg_gate `
  --checkpoint-path checkpoints/ssg_gate_best.pt
```

### 训练参数说明

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--data-path` | `data/raw/RML2016.10a_dict.pkl` | RadioML2016.10A 原始 pickle 文件 |
| `--split-dir` | `data/splits` | train/val/test 索引和 `split_meta.json` 所在目录 |
| `--model` | `ssg_gate` | 模型名称 |
| `--epochs` | `50` | 训练轮数 |
| `--batch-size` | `256` | batch size |
| `--lr` | `1e-3` | AdamW 学习率 |
| `--weight-decay` | `1e-4` | AdamW 权重衰减 |
| `--dropout` | `0.3` | 分类头 dropout |
| `--feature-dim` | `64` | 每个 CNN 分支输出特征维度 |
| `--structure-alpha` | `0.2` | SSG gate 中结构分数的权重系数，只影响 `ssg_gate` 和 `ssg_gated_concat` |
| `--seed` | `42` | 随机种子 |
| `--device` | `auto` | `auto`、`cuda` 或 `cpu` |
| `--results-dir` | `results/multiview/ssg_gate` | 保存 `config.json` 和 `train_log.csv` 的目录 |
| `--checkpoint-path` | `checkpoints/ssg_gate_best.pt` | best checkpoint 保存路径 |
| `--fft-shift` / `--no-fft-shift` | `--fft-shift` | 是否对 FFT view 使用 `fftshift` |
| `--fft-transform` | `log1p` | FFT 幅度变换，可选 `log1p`、`standardize`、`log1p_standardize`、`none` |
| `--num-workers` | `0` | PyTorch DataLoader worker 数 |
| `--max-train-samples` | 无 | 只取前 N 个训练样本，主要用于 CPU 冒烟测试 |
| `--max-val-samples` | 无 | 只取前 N 个验证样本，主要用于 CPU 冒烟测试 |

### `--structure-alpha` 说明

`--structure-alpha` 只对 `ssg_gate` 和 `ssg_gated_concat` 生效，用来控制 signal-structure descriptors 对 gate score 的影响强度。

SSG gate 中每个视图的 score 计算方式为：

```text
score_v = feat_score_v + structure_alpha * struct_score_v
```

其中：

```text
feat_score_v    来自 CNN 分支提取的深度特征 z_v
struct_score_v  来自结构指标 q_v
```

默认值为：

```powershell
--structure-alpha 0.2
```

这个设置表示：训练初期仍然主要依赖深度特征产生 gate 权重，同时允许结构指标逐渐参与视图选择。代码中还对结构分数 MLP 的最后一层做了 zero initialization，所以 `ssg_gate` 和 `ssg_gated_concat` 在初始化时会更接近 `vanilla_gate`，随后再通过训练学习如何利用结构指标。

常见实验取值可以从下面几组开始：

```text
0.0   结构分支被关闭，接近只使用 feature gate
0.1   较弱结构引导
0.2   默认设置，较稳妥
0.5   更强结构引导
1.0   结构分数和特征分数同等尺度相加
```

### 训练输出

训练完成后会保存：

```text
results/multiview/ssg_gate/config.json
results/multiview/ssg_gate/train_log.csv
checkpoints/ssg_gate_best.pt
```

`train_log.csv` 字段包括：

```text
epoch, train_loss, train_acc, val_loss, val_acc
```

## 评估脚本

脚本路径：

```powershell
scripts/evaluate_multiview.py
```

推荐测试集评估命令：

```powershell
python scripts/evaluate_multiview.py `
  --checkpoint-path checkpoints/ssg_gate_best.pt `
  --model ssg_gate `
  --data-path data/raw/RML2016.10a_dict.pkl `
  --split-dir data/splits `
  --split test `
  --batch-size 256 `
  --device auto `
  --results-dir results/multiview/ssg_gate_test
```

### 评估参数说明

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--checkpoint-path` | 必填 | 训练得到的 checkpoint |
| `--model` | checkpoint 中的模型名 | 模型名称 |
| `--data-path` | `data/raw/RML2016.10a_dict.pkl` | RadioML2016.10A 原始 pickle 文件 |
| `--split-dir` | `data/splits` | 划分目录 |
| `--split` | `test` | 评估 `test` 或 `val` |
| `--results-dir` | `results/multiview/eval` | 评估结果输出目录 |
| `--batch-size` | `256` | batch size |
| `--device` | `auto` | `auto`、`cuda` 或 `cpu` |
| `--feature-dim` | checkpoint 配置或 `64` | 特征维度 |
| `--dropout` | checkpoint 配置或 `0.3` | dropout |
| `--structure-alpha` | checkpoint 配置或 `0.2` | SSG gate 中结构分数的权重系数，评估时通常保持和训练 checkpoint 一致 |
| `--fft-shift` / `--no-fft-shift` | checkpoint 配置或 `True` | 是否使用 `fftshift` |
| `--fft-transform` | checkpoint 配置或 `log1p` | FFT 幅度变换 |
| `--num-workers` | `0` | DataLoader worker 数 |
| `--max-samples` | 无 | 只评估前 N 个样本，主要用于 CPU 冒烟测试 |

### 评估输出

评估脚本会生成：

```text
overall_metrics.json
predictions.csv
accuracy_by_snr.csv
accuracy_by_modulation.csv
accuracy_by_modulation_snr.csv
confusion_matrix.csv
```

其中 `predictions.csv` 至少包含：

```text
sample_id, true_label, pred_label, true_modulation, pred_modulation, snr, correct
```

如果模型是 `vanilla_gate`、`ssg_gate` 或 `ssg_gated_concat`，还会额外生成：

```text
gate_weights.csv
gate_weights_by_snr.csv
gate_weights_by_modulation.csv
```

`gate_weights.csv` 字段包括：

```text
sample_id, modulation, snr, w_iq, w_ap, w_fft, correct
```

## CPU 小样本验证

如果只想确认代码能跑通，可以使用小样本命令：

```powershell
python scripts/train_multiview.py `
  --model ssg_gate `
  --epochs 1 `
  --batch-size 128 `
  --device cpu `
  --structure-alpha 0.2 `
  --max-train-samples 512 `
  --max-val-samples 256 `
  --results-dir results/multiview/smoke_ssg_gate `
  --checkpoint-path checkpoints/smoke_ssg_gate_best.pt
```

然后评估：

```powershell
python scripts/evaluate_multiview.py `
  --checkpoint-path checkpoints/smoke_ssg_gate_best.pt `
  --model ssg_gate `
  --split val `
  --batch-size 128 `
  --device cpu `
  --max-samples 256 `
  --results-dir results/multiview/smoke_ssg_gate_eval
```

## 推荐第一组实验

建议先跑完整方法和几个关键消融：

```powershell
python scripts/train_multiview.py --model ssg_gate --structure-alpha 0.2 --epochs 50 --batch-size 256 --device auto --results-dir results/multiview/ssg_gate --checkpoint-path checkpoints/ssg_gate_best.pt
python scripts/train_multiview.py --model ssg_gated_concat --structure-alpha 0.2 --epochs 50 --batch-size 256 --device auto --results-dir results/multiview/ssg_gated_concat --checkpoint-path checkpoints/ssg_gated_concat_best.pt
python scripts/train_multiview.py --model vanilla_gate --epochs 50 --batch-size 256 --device auto --results-dir results/multiview/vanilla_gate --checkpoint-path checkpoints/vanilla_gate_best.pt
python scripts/train_multiview.py --model concat --epochs 50 --batch-size 256 --device auto --results-dir results/multiview/concat --checkpoint-path checkpoints/concat_best.pt
```

然后分别在 test split 上评估：

```powershell
python scripts/evaluate_multiview.py --checkpoint-path checkpoints/ssg_gate_best.pt --model ssg_gate --split test --results-dir results/multiview/ssg_gate_test
python scripts/evaluate_multiview.py --checkpoint-path checkpoints/ssg_gated_concat_best.pt --model ssg_gated_concat --split test --results-dir results/multiview/ssg_gated_concat_test
python scripts/evaluate_multiview.py --checkpoint-path checkpoints/vanilla_gate_best.pt --model vanilla_gate --split test --results-dir results/multiview/vanilla_gate_test
python scripts/evaluate_multiview.py --checkpoint-path checkpoints/concat_best.pt --model concat --split test --results-dir results/multiview/concat_test
```

## 数据加载方式说明

数据加载代码位于 `mvnet/radioml.py`。

当前 Dataset 不是把所有样本预先拼成一个巨大的 `[220000, 2, 128]` 数组，而是：

1. 读取 `split_meta.json`，获得全局 sample_id 的定义方式、类别映射、SNR 顺序和每个 `(modulation, snr)` 组的起止区间。
2. 读取 `train_indices.npy`、`val_indices.npy` 或 `test_indices.npy`，得到当前 split 需要访问的全局 sample_id。
3. 加载原始 `RML2016.10a_dict.pkl`，保留原始字典结构。
4. 在 `__getitem__` 中，根据 sample_id 查找它属于哪个 `(modulation, snr)` 组。
5. 计算该样本在组内的 local index。
6. 从原始字典中取出对应 IQ 样本。
7. 即时构造 IQ/AP/FFT 三个 view，并返回给 PyTorch DataLoader。

返回的 batch 字段包括：

```text
iq         [B, 2, 128]
ap         [B, 2, 128]
fft        [B, 1, 128]
label      [B]
sample_id  [B]
snr        [B]
```

这种写法的好处是不会额外复制一份完整大数组，和 `split_meta.json` 中定义的全局索引顺序保持一致。需要注意的是，Windows 下如果把 `--num-workers` 设置得很大，每个 worker 可能会各自持有一份 pickle 数据，内存占用会明显增加；当前默认 `--num-workers 0` 是更稳妥的选择。
