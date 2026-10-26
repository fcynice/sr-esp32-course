# ESP32 Camera Degradation (versioned, offline)

基于真实 `data/esp32_camera/data01` 校准的 ESP32 摄像头退化模型。当前流程是
**离线生成带版本、带变体的 LR 数据集**，再交给原有配对训练管线使用。

## 版本与文件

```text
src/superresolution/degrade/
├── __init__.py   # 版本注册、degrade_image、代码指纹
├── ops.py        # 底层算子（随版本冻结）
├── v1.py         # v1 配方（已冻结）
├── v2.py         # v2 配方：削弱 bloom / ghosting，修复高光黑圈
├── stats.py      # 模拟图 / 参考图统计
├── cli.py        # LR 数据集生成 CLI
└── __main__.py   # python -m superresolution.degrade
```

| 版本 | 说明 | 指纹 |
|---|---|---|
| `v1` | 初始审核版本，保留完整 bloom / ghosting | `555a9e08e8de9a57cc49b50564ddc8211cc8bff72bf972318b4150190c05ee9a` |
| `v2` | 默认版本；`overexposed` bloom 从 `0.15–0.55` 降到 `0–0.08`，ghosting 强度/位移/拖影减弱 | `4bee71cbf19db7a48ddc28232d096c311833a29962684187afbc0b78ceb4f28c` |

版本冻结规则：

- `v1.py`、`v2.py` 审核后不再修改。
- 后续调参新增 `v3.py` 并在 `__init__.py` 注册。
- 共享的 `ops.py` 也参与指纹；新版本如需不同底算子，应新增独立模块。
- manifest 和训练 `run_info.json` 记录指纹，旧数据始终可追溯。

## 数据管理方式

LR 目录按 **版本 + 变体** 分层，方便识别和长期并存：

```text
data/
├── DIV2K/
│   ├── DIV2K_train_LR_esp32_v1/X4/v0/0001x4.png
│   ├── DIV2K_train_LR_esp32_v2/X4/v0/0001x4.png
│   ├── DIV2K_train_LR_esp32_v2/X4/v1/0001x4.png
│   ├── DIV2K_valid_LR_esp32_v1/X4/v0/0801x4.png
│   └── DIV2K_valid_LR_esp32_v2/X4/v0/0801x4.png
└── Flickr2K/
    ├── Flickr2K_LR_esp32_v1/X4/v0/000001x4.png
    ├── Flickr2K_LR_esp32_v2/X4/v0/000001x4.png
    └── Flickr2K_LR_esp32_v2/X4/v1/000001x4.png
```

规则：

- `LR_esp32_vN`：退化模型版本
- `X4`：放大倍数
- `v0/`、`v1/`：同一 HR 的不同随机退化变体
- 文件名保持 `<HR stem>x4.<ext>`，与原有 HR/LR 对齐规则一致
- manifest 也按版本隔离：

```text
data/esp32_degrade_dataset/
├── v1/{manifest.json,manifest.jsonl,summary.json}
└── v2/{manifest.json,manifest.jsonl,summary.json}
```

训练时 `--lr-type esp32_v2` 会读取 `LR_esp32_v2/X4`，并把 `v0`、`v1` 都识别为
同一 HR 的候选变体，每次取 patch 时随机选一个。验证固定使用 `v0`，保证不同
epoch 的 PSNR 可比。

## 退化覆盖

- **噪声**：散粒噪声、读出噪声、色度噪声、行列固定模式噪声、热像素
- **色彩/曝光**：绿偏、白平衡误差、低饱和、欠曝、黑位抬升、低对比、gamma
- **镜头**：高斯软焦、失焦、运动模糊、色差、暗角
- **时间域**：多帧残影、慢快门拖影、滚动快门条带
- **高光**：soft highlight clipping；v2 中 bloom 已大幅削弱
- **压缩**：真实 JPEG 编解码（4:2:0 / 4:2:2 / 4:4:4）
- **配方**：9 个 profile，训练 split 按权重随机采样

| Profile | 权重 | 主要内容 |
|---|---:|---|
| `green_indoor` | 0.20 | 荧光灯绿偏、低饱和、软焦、弱光 |
| `lowlight_noise` | 0.20 | 欠曝、暗部强噪声、色度噪声、暗角 |
| `motion_blur` | 0.12 | 运动模糊、低对比 |
| `ghosting` | 0.12 | 残影、拖影、条带、轻微滚动快门 |
| `defocus` | 0.12 | 失焦、镜头软、低对比 |
| `day_soft` | 0.10 | 白天、轻软焦、轻噪声 |
| `overexposed` | 0.07 | 高光裁切、轻 bloom、雾化 |
| `jpeg_hard` | 0.05 | 低质量 JPEG、块效应、色带 |
| `mixed_hard` | 0.02 | 多种重度退化叠加 |

## 生成 LR 数据集

```bash
# 默认 v2 + 每张 HR 两个变体
python -m superresolution.degrade --version v2 --variants 2 --scale 4

# 只生成一小批做检查
python -m superresolution.degrade --version v2 --splits div2k_valid --max-items 10

# 只启用部分 profile
python -m superresolution.degrade --profiles green_indoor,lowlight_noise

# 先生成 v1 单变体（历史版本）
python -m superresolution.degrade --version v1 --variants 1
```

主要参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--data-root` | `data` | 数据根目录 |
| `--scale` | `4` | 降采样倍数，对应 `X4` |
| `--version` | `v2` | 冻结退化版本 |
| `--variants` | `2` | 每个 HR 的 LR 变体数量，写入 `v0..vN` |
| `--severity` | `1.0` | 全局退化强度 |
| `--seed` | `42` | 采样种子；每张图/每个变体使用稳定 hash |
| `--splits` | 全部 | `div2k_train,div2k_valid,flickr2k` |
| `--profiles` | 全部 | profile 子集 |
| `--profile-mode` | `auto` | 训练 split 加权随机，DIV2K valid 轮转覆盖 |
| `--output-format` | `png` | `png` / `jpg` |
| `--max-items` | `0` | 每个 split 限制 HR 数量，调试用 |
| `--workers` | `4` | 并行进程数 |
| `--overwrite` | — | 覆盖已存在文件；默认跳过 |
| `--manifest-dir` | `data/esp32_degrade_dataset` | 会自动追加 `/<version>` |
| `--no-manifest` | — | 不写 manifest |
| `--dry-run` | — | 只打印数量和输出目录 |

## 训练

```bash
# 使用 v2 多变体
python -m superresolution.train \
  --model edsr --lr-type esp32_v2 \
  --patch-size 192 --batch-size 16 --epochs 80
```

用已有 run 初始化模型权重：

```bash
python -m superresolution.train \
  --model edsr --lr-type esp32_v2 \
  --init-from runs/res05/best.pth \
  --lr 1e-4 --epochs 60 \
  --patch-size 192 --batch-size 16
```

训练行为：

- `collect_hr_items` 会把同一 HR 的 `v0`、`v1` 都收集起来
- `SRMixTrainDataset` 每次取 patch 时随机选一个变体
- 每个 epoch 仍然是每张 HR 一个样本，但不同 epoch 会看到不同变体
- 验证固定用 `v0`，PSNR 在不同 epoch 间保持可比
- `--lr-type esp32_v1` 仍可读取旧的 v1 单变体数据

## 注意

- 当前 v1 数据已迁移到 `LR_esp32_v1/X4/v0`；`--lr-type esp32_v1` 可继续使用。
- 预生成 LR 后，训练数据是固定的；多 variant 提供随机性。
- v2 主要修复 v1 训练中高光周围出现的黑圈：黑圈来自模型对 `bloom` 的反演，
  不是原图问题。
- LR 退化很重，PSNR 不能与 bicubic / unknown 的绝对值直接比较；应与
  “退化 LR 直接 bicubic 放大”的基线对比。
- 历史的人工审查可视化服务仍在 `servers/degrade_review_server/`；当前生成器不再
  产生新的 review session。
