# ESP32 Camera Degradation (versioned, offline)

基于真实 `data/esp32_camera/data01` 校准的 ESP32 摄像头退化模型。当前流程是
**离线生成 LR 数据集**，再交给原有配对训练管线使用；不再在训练时实时退化。

## 文件

```text
src/superresolution/degrade/
├── __init__.py   # 版本注册、degrade_image、代码指纹
├── ops.py        # 底层算子（模糊/噪声/残影/JPEG/色调等，随版本冻结）
├── v1.py         # v1 配方与管线（审核后冻结）
├── stats.py      # 模拟图 / 参考图统计
├── cli.py        # ESP32 LR 数据集生成 CLI
└── __main__.py   # python -m superresolution.degrade
```

## 版本冻结与代码指纹

- `v1.py` 的 profile 范围和管线在人工审核后视为冻结。
- 后续调参必须新增 `v2.py`、`v3.py` 并在 `__init__.py` 注册，**不要修改 `v1.py`**。
- 共享的 `ops.py` 也参与版本指纹；新版本如需不同底算子，应新增独立模块。
- 生成 manifest 和训练 `run_info.json` 都记录 `fingerprint`，用于确认数据 / run
  对应的退化代码没有被改过。
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
- **配方**：9 个 profile，训练 split 按权重随机采样

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

## 生成 LR 数据集

```bash
# 生成全部三个 split，默认 v1 / scale 4 / PNG / 每张 HR 一个变体
python -m superresolution.degrade --version v1 --scale 4

# 只生成一小批做检查
python -m superresolution.degrade --splits div2k_valid --max-items 10

# 只启用部分 profile
python -m superresolution.degrade --profiles green_indoor,lowlight_noise

# 输出 JPEG（默认 PNG；PNG 不会对已烘焙的 JPEG 痕迹二次压缩）
python -m superresolution.degrade --output-format jpg
```

输出目录与 `train.py` 的配对规则一致：

```text
data/
├── DIV2K/
│   ├── DIV2K_train_LR_esp32/X4/<HR stem>x4.png
│   └── DIV2K_valid_LR_esp32/X4/<HR stem>x4.png
└── Flickr2K/
    └── Flickr2K_LR_esp32/X4/<HR stem>x4.png
```

默认还会写：

```text
data/esp32_degrade_dataset/
├── manifest.json      # 版本、指纹、split 数量、输出目录
├── manifest.jsonl     # 每张图：源 HR、LR、profile、seed、参数、统计
└── summary.json       # overall / by_profile / by_split 统计
```

主要参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--data-root` | `data` | 数据根目录 |
| `--scale` | `4` | 降采样倍数，对应 `X4` 目录 |
| `--version` | `v1` | 冻结退化模型版本 |
| `--severity` | `1.0` | 全局退化强度 |
| `--seed` | `42` | 采样种子；每张图还会混入源路径 hash |
| `--splits` | 全部 | `div2k_train,div2k_valid,flickr2k` |
| `--profiles` | 全部 | profile 子集 |
| `--profile-mode` | `auto` | 训练 split 加权随机，DIV2K valid 轮转覆盖 |
| `--output-format` | `png` | `png` / `jpg` |
| `--max-items` | `0` | 每个 split 限制数量，调试用 |
| `--workers` | `4` | 并行进程数 |
| `--overwrite` | — | 覆盖已存在文件；默认跳过 |
| `--manifest-dir` | `data/esp32_degrade_dataset` | manifest / summary 目录 |
| `--no-manifest` | — | 不写 manifest |
| `--dry-run` | — | 只打印数量和输出目录 |

已存在的目标文件默认跳过。因此做全量生成时先跑：

```bash
python -m superresolution.degrade --dry-run
```

## 训练

`--lr-type esp32` 现在和 `bicubic` / `unknown` 一样走原有配对数据集：

```bash
python -m superresolution.train \
  --model edsr --lr-type esp32 \
  --patch-size 192 --batch-size 16 --epochs 150
```

训练时：

- 从 `DIV2K_train_LR_esp32/X4` / `Flickr2K_LR_esp32/X4` 读取预生成 LR
- 使用原有 `SRMixTrainDataset` 随机对齐裁 `(LR patch, HR patch)`
- 不再解码整图后实时退化，不再在 worker 中做 JPEG / ghosting / noise
- 每个 epoch 每张 HR 一个样本；Flickr2K 仍按 HR 做 95/5 划分

验证使用 `SRValDataset` 读取预生成 LR，整图计算 PSNR；退化在生成阶段已经固定，
因此不同 epoch 的 PSNR 可直接比较。

## 注意

- 当前**每张 HR 生成一个 LR 变体**。多 variant 需要扩展 `collect_hr_items` /
  dataset 的随机选择逻辑，后续再加。
- 预生成 LR 后，训练数据是固定的；如果发现对特定噪声/JPEG 过拟合，再生成多个
  变体或换新的退化版本。
- 由于 LR 退化很重，PSNR 不能与 bicubic / unknown 的绝对值直接比较；应与
  “退化 LR 直接 bicubic 放大”的基线对比。
- 历史的人工审查可视化服务仍在 `servers/degrade_review_server/`；当前生成器不再
  产生新的 review session。
