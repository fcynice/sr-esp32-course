# ESP32 Camera Degradation (versioned)

基于真实 `data/esp32_camera/data01` 校准的 ESP32 摄像头退化模型。训练时按 patch
实时生成 LR，不预生成全量 LR 数据集。

## 文件

```text
src/superresolution/degrade/
├── __init__.py   # 版本注册、公共 API、代码指纹
├── ops.py        # 底层算子（模糊/噪声/残影/JPEG/色调等，随版本冻结）
├── v1.py         # v1 配方与管线（审核后冻结）
├── stats.py      # 模拟图 / 参考图统计
├── cli.py        # 人工审查图生成 CLI
└── __main__.py   # python -m superresolution.degrade
```

训练侧使用 `dataset.py` 中的 `ESP32SRTrainDataset` / `ESP32SRValDataset`，由
`train.py --lr-type esp32` 接入。

## 版本冻结与代码指纹

- `v1.py` 的 profile 范围和管线在人工审核后视为冻结。
- 后续调参必须新增 `v2.py`、`v3.py` 并在 `__init__.py` 注册，**不要修改 `v1.py`**。
- 共享的 `ops.py` 也参与版本指纹；新版本如需不同底算子，应新增独立模块。
- 审查 `manifest.json` 和训练 `run_info.json` 都记录 `fingerprint`，用于确认旧
  session / checkpoint 对应的退化代码没有被改过。
- 当前 v1 指纹：

```text
555a9e08e8de9a57cc49b50564ddc8211cc8bff72bf972318b4150190c05ee9a
```

## 退化覆盖

- **噪声**：散粒噪声、读出噪声、色度噪声、行列固定模式噪声、热像素
- **色彩/曝光**：绿偏、白平衡误差、低饱和、欠曝、黑位抬升、低对比、gamma
- **镜头**：高斯软焦、失焦、运动模糊、色差、暗角
- **时间域**：多帧残影、慢快门拖影、滚动快门条带
- **高光**：软裁切、灯光泛光
- **压缩**：真实 JPEG 编解码（4:2:0 / 4:2:2 / 4:4:4，按 quality 自动选择）
- **配方**：9 个 profile，训练时按权重随机采样

| Profile | 权重 | 主要内容 |
|---|---:|---|
| `green_indoor` | 0.20 | 荧光灯绿偏、低饱和、软焦、弱光 |
| `lowlight_noise` | 0.20 | 欠曝、暗部强噪声、色度噪声、暗角 |
| `motion_blur` | 0.12 | 运动模糊、低对比 |
| `ghosting` | 0.12 | 多帧残影、拖影、条带、轻微滚动快门 |
| `defocus` | 0.12 | 失焦、镜头软、低对比 |
| `day_soft` | 0.10 | 白天、轻软焦、轻噪声、较正常色彩 |
| `overexposed` | 0.07 | 高光裁切、泛光、雾化 |
| `jpeg_hard` | 0.05 | 低质量 JPEG、块效应、色带 |
| `mixed_hard` | 0.02 | 多种重度退化叠加，困难样本 |

## 训练数据划分（`--lr-type esp32`）

只使用 HR，不读取任何 `*_LR_*` 目录：

| 用途 | 来源 | 数量 |
|---|---|---:|
| 训练 | DIV2K train 全部 | 800 |
| 训练 | Flickr2K 的 95% | 2518 |
| 验证 | DIV2K valid 全部 | 100 |
| 验证 | Flickr2K 的 5% | 132 |
| **训练合计** |  | **3318 HR / epoch** |
| **验证合计** |  | **232 HR** |

Flickr2K 的 5% 验证集按 HR 图像划分，同一 HR 不会同时出现在训练和验证中。
划分由 `--seed` 控制。

## 训练样本构造

默认 `scale=4`、`patch_size=192`、`degrade_margin=8` LR px。每次
`__getitem__` 都重新随机：

```text
随机 HR
-> 随机裁 HR 区域 256×256       # patch_size + 2 * margin * scale
-> 整块实时退化并降采样        # 256×256 -> LR 64×64
-> 中心裁掉 margin             # LR 64×64 -> 48×48
-> 同时裁 HR 中心              # HR 256×256 -> 192×192
-> HR/LR 同步翻转/90° 旋转
-> 返回 (LR 48×48, HR 192×192)
```

带 margin 的目的是让模糊、残影、暗角、JPEG 在 patch 边缘自然过渡，避免逐
patch 独立退化产生明显拼接边界。

训练时每次访问都会重新采样 **profile + 退化参数 seed + patch 位置**，因此：

- 同一张 HR 在不同 epoch 看到不同 patch 和不同退化
- 每个 epoch 每张 HR 只出现一次：3318 个训练样本
- profile 按上表权重随机；可用 `--degrade-profiles` 限制子集
- `--degrade-severity` 控制整体退化强度

训练循环沿用现有训练器：`shuffle=True`、默认 `batch_size=16`、
`drop_last=True`、`num_workers=4`、L1 loss、Adam、CosineAnnealing、AMP。

## 验证策略

验证集使用固定退化，保证不同 epoch 的 PSNR 可比：

```text
第 i 张验证 HR：
  profile = profiles[i % len(profiles)]
  seed    = args.seed + i * 1000003
  HR 整图 -> ESP32 退化 -> 整图 LR
  HR 裁到 LR 尺寸 × scale
  计算整图 PSNR
```

- 232 张验证图，9 个 profile 大致均匀轮转
- `val_psnr` 是这些图 PSNR 的平均值，不是按训练 profile 权重加权
- 该 PSNR **不能**与 bicubic / unknown 的 PSNR 绝对值直接比较；LR 退化更重，
  绝对值会低很多，主要用于看训练趋势和相对提升

## 人工审查图

```bash
# 9 个 profile × (DIV2K, Flickr2K) × 每个 2 张 = 36 张
python -m superresolution.degrade --per-profile 2 --seed 42
```

输出到 `data/esp32_degrade_review/<version>_<时间戳>/`，包含 `manifest.jsonl`、
`summary.json`、模拟 LR 图片和 contact sheet。审查可视化的启动方式、页面功能和
接口说明见 [`servers/degrade_review_server/README.md`](../../../servers/degrade_review_server/README.md)。

## 训练命令

```bash
python -m superresolution.train \
  --model edsr --lr-type esp32 --degrade-version v1 \
  --patch-size 192 --batch-size 16 --epochs 150
```

调试：

```bash
python -m superresolution.train \
  --model espcn --lr-type esp32 \
  --degrade-profiles green_indoor,ghosting \
  --max-train-items 8 --max-val-items 4 \
  --patch-size 64 --batch-size 2 --epochs 1 \
  --val-every 1 --num-workers 0 --device cpu
```

## 注意

- `--lr-type esp32` 不会和 bicubic / unknown 混合；`both` 仍表示原来的两种 LR。
- `--lr-type esp32` 完全跳过 `build_datasets()`，因此不要求任何预生成 LR 目录。
- 训练时 JPEG 压缩作用在带 margin 的 LR patch 上；验证时作用在整图 LR 上。
- 本轮未处理 `esp_video` 抽帧；视频参考留作后续可选扩展。
