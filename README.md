# Super-Resolution (ESPCN / EDSR + PixelShuffle)

基于 PyTorch 的图像超分辨率项目，提供两种结构（均使用亚像素上采样 `nn.PixelShuffle`），在 DIV2K 与 Flickr2K 上训练：

- **`edsr`（默认，推荐）**：残差网络 + PixelShuffle，默认 128 通道 / 32 残差块，约 **10.78M** 参数。
- **`espcn`**：3 个卷积层 + PixelShuffle，约 37K 参数；保留用于兼容旧的 res01/res02 checkpoint。

## 网络结构

### ESPCN

RGB 输入、放大倍数 `r`（默认 4）：

| 层 | 卷积核 | 通道 | 激活 |
|---|---|---|---|
| Conv1 | 5×5 | 3 → 64 | ReLU |
| Conv2 | 3×3 | 64 → 32 | ReLU |
| Conv3 | 3×3 | 32 → 3·r² | 无 |
| PixelShuffle(r) | — | 3·r² → 3，同时 H×r, W×r | — |

- 最后一层不接激活，输出 `3·r²` 个通道，由 `PixelShuffle` 重排为 `r×r` 的空间块，完成上采样。
- X4 时 48 通道 → 3 通道，参数量约 37K。
- 通道数可通过 `--mid1` / `--mid2` 调整。

### EDSR-lite（推荐）

特征提取全部在低分辨率空间完成，最后再上采样：

```
head  Conv3x3        3 -> C
body  N x ResidualBlock(C)          # x + 0.1 * conv2(relu(conv1(x)))
fuse  conv_after(body) + head       # 全局残差（LR 空间）
up    [Conv3x3 C -> C*4, PixelShuffle(2)] x log2(r)   # r=2/4/8
      (r=3 时改用一级 Conv3x3 C -> C*9, PixelShuffle(3))
tail  Conv3x3        C -> 3
```

- 无 BatchNorm；残差分支乘 `0.1`，避免深层堆叠时数值爆炸。
- 上采样：r=2/4/8 用多级 `PixelShuffle(2)`；r=3 用单个 `PixelShuffle(3)`。
- 默认 `C=128, N=32` → **10.78M** 参数；可用 `--channels` / `--n-blocks` 调整。
- 显存实测（RTX 5070 8GB，patch 192、batch 16）：训练峰值约 **1.2 GB**，整图验证约 1.8 GB。

### 模型对比（bicubic X4，实测）

| 模型 | 参数量 | 5 epoch PSNR | 备注 |
|---|---|---|---|
| `espcn` | 37K | 26.2 dB | 100 epoch 收敛到 27.6 dB |
| `edsr` 128/32 | 10.8M | **27.3 dB** | 仍在上升，收敛更快 |

## 数据准备

目录结构（`data/` 已被 `.gitignore` 忽略）：

```
data/
├── DIV2K/
│   ├── DIV2K_train_HR/                 # 800 张 HR (.jpeg)
│   ├── DIV2K_valid_HR/                 # 100 张 HR (.jpeg)
│   ├── DIV2K_train_LR_bicubic/X4/      # <stem>x4.png
│   ├── DIV2K_train_LR_unknown/X4/      # <stem>x4.png
│   └── DIV2K_valid_LR_*/X4/
└── Flickr2K/
    ├── Flickr2K_HR/                    # 2650 张 HR (.png)
    ├── Flickr2K_LR_bicubic/X4/         # <stem>x4.png
    └── Flickr2K_LR_unknown/X4/         # <stem>x4.png
```

> 目录中的 `X4` 随 `--scale` 变化（`X2` / `X3` / `X4`）。LR 默认按 `<HR stem>x{scale}.png`
> 匹配（例如 HR `0001.jpeg` ↔ LR `0001x4.png`，其他扩展名也可）。接入自己的数据集时
> 按同样结构放置，或修改 `train.py` 顶部的 `TRAIN_SOURCES` / `VAL_SOURCES`。

下载脚本（可选，数据已就位可跳过）：

```bash
bash scripts/download_div2k.sh       # 下载 DIV2K
bash scripts/download_div2k_lr.sh    # 下载 DIV2K 的 LR 归档 (bicubic + unknown)
```

> 训练默认 **`--lr-type both`**：每张 HR 同时对应 bicubic 和 unknown 两套 LR，
> 每轮每张 HR 采样 2 次，每次**随机**选一种 LR。因此同一张 HR 在一轮内可能分别遇到两种 LR，
> 也可能两次都是同一种（不是必然）。
> 也可用 `--lr-type bicubic` 或 `--lr-type unknown` 只训练单一退化类型。
> LR 与 HR 严格整数倍对齐（HR = r × LR），dataset 切割时按同一空间位置对齐取块。
>
> 验证默认覆盖训练所用的全部 LR 类型（`both` 时同时报告 bicubic / unknown 的 PSNR），
> 控制台会显示分项：`val_psnr=... (bicubic=..., unknown=...)`。
>
> 新增 `--lr-type esp32`：先用 `python -m superresolution.degrade` 把退化 LR 生成到
> `DIV2K_train_LR_esp32/X4` 等目录，再像 `bicubic` / `unknown` 一样走原有对齐裁剪训练。
> 数据划分仍为 DIV2K train + Flickr2K 95% 训练、DIV2K valid + Flickr2K 5% 验证。
> 详见 [degrade/README.md](src/superresolution/degrade/README.md)。

### bicubic vs unknown（实测参考）

X4、相同网络与数据划分下的验证 PSNR：

| LR 类型 | 训练量 | 验证 PSNR |
|---|---|---|
| `bicubic` | 3 epoch | **25.7 dB（仍在上升）** |
| `unknown` | 42 epoch | 22.9 dB |

`unknown` 的 LR 是重度模糊退化（其到 HR 的 MAE 约为 bicubic LR 的 2 倍），3 层小网络难以完全还原；
即使对比 bicubic 上采样基线，模型也仅领先约 +0.9 dB。建议先用 `bicubic` 跑通，再尝试 `unknown` 或更大的网络。

> 上表是早期单类型实验的参考。当前默认 `--lr-type both` 会同时训练两种退化；
> 只想复现单类型时用 `--lr-type bicubic` 或 `--lr-type unknown`。

### 数据划分

| 用途 | 来源 | HR 图片数 | 单类型样本 | `both` 样本 |
|---|---|---|---|---|
| 训练 | DIV2K train 全部 + Flickr2K 的 95% | 800 + 2518 = 3318 | 3318 | 6636 |
| 验证 | DIV2K valid 全部 + Flickr2K 的 5% | 100 + 132 = 232 | 232 | 464 |

- Flickr 的 5% 验证集按 **HR 图片** 划分（不是按 LR 对），因此同一张 HR 不会同时出现在训练和验证中。
- `--lr-type both` 时每张 HR 暴露 2 个样本，所以样本数和单 epoch 时间约翻倍。
- 验证在**整图**上计算 PSNR（标准 SR 评价协议）。`both` 时分别验证两种 LR；`train_log.csv` 的 `val_psnr` 是所有验证图（两种 LR 合计）的平均值，`best.pth` 也按这个平均值选取。
- 划分使用固定随机种子（默认 `--seed 42`），保证可复现。

## 安装

```bash
uv sync        # 或已就绪的 .venv
```

依赖：`torch`、`torchvision`、`tqdm`（GPU 版通过 `pytorch-cu130` 索引安装）。

## 训练

```bash
# 推荐：EDSR 128 通道 / 32 残差块（默认配置）
.venv/bin/python -m superresolution.train \
  --model edsr --patch-size 192 --batch-size 16 --epochs 150

# 轻量：ESPCN
.venv/bin/python -m superresolution.train \
  --model espcn --patch-size 128 --epochs 100
```

或使用安装的命令：

```bash
train-sr --model edsr --epochs 150
```

主要参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--data-root` | `data` | 数据集根目录 |
| `--output-dir` | `runs` | 输出目录 |
| `--model` | `edsr` | `espcn` 或 `edsr` |
| `--scale` | `4` | 放大倍数（2/3/4/8） |
| `--lr-type` | `both` | `bicubic` / `unknown` / `both` / `esp32`（读取预生成 ESP32 LR） |
| `--patch-size` | `192` | HR 训练块大小（须能被 scale 整除，X4 对应 LR 48×48） |
| `--batch-size` | `16` | 批大小 |
| `--epochs` | `100` | 训练轮数 |
| `--lr` | `2e-4`(edsr) / `1e-3`(espcn) | 初始学习率（Adam + CosineAnnealing） |
| `--channels` | `128` | EDSR body 通道数 |
| `--n-blocks` | `32` | EDSR 残差块数 |
| `--mid1` / `--mid2` | `64` / `32` | ESPCN 中间层通道数 |
| `--val-every` | `5` | 每多少 epoch 验证一次 |
| `--num-workers` | `4` | DataLoader 线程数 |
| `--no-augment` | — | 关闭翻转/旋转增强 |
| `--no-amp` | — | 关闭混合精度 |
| `--seed` | `42` | 随机种子（数据划分 / 增强 / LR 采样） |
| `--init-from` | — | 从 checkpoint / run 目录只加载模型权重（不加载 optimizer/scheduler/epoch） |
| `--resume` | — | 从 run 目录或 checkpoint 续训（见「断点续训」） |

> `--max-train-items` / `--max-val-items` 是调试参数：前者限制训练的 **HR 图片数**（`both` 时实际样本为 2 倍），后者限制展平后的验证样本数。

训练产物保存在 `runs/res{序号}/`，序号按时间顺序递增（越早越小，两位补零）：

```
runs/
├── res01/          # 最早的运行
└── res02/
```

每个运行目录包含：

- `best.pth` — 验证 PSNR 最优权重
- `last.pth` — 最新权重
- `train_log.csv` — 每轮 loss / PSNR / 耗时
- `last_val.txt` — 最近一次验证每张图的 PSNR
- `args.json` — 本次运行的完整参数
- `run_info.json` — 序号、时间戳、模型、scale、`lr_type` 等元信息

目录名只保留 `resNN`，模型/放大倍数/LR 类型等信息都在 `run_info.json`（与 `args.json`）里。

checkpoint（`best.pth` / `last.pth`）是**自描述**的，内部保存了 `model_name` 与 `model_kwargs`，推理时自动重建对应网络结构（ESPCN 或 EDSR）。**旧 checkpoint（如 res01/res02）没有这些字段，会自动回退为 ESPCN**，仍然可以正常推理。

> 旧的 `espcn_x4_{时间戳}` 或 `espcn_x4_unknown_01` 目录可用 `scripts/migrate_runs.py` 迁移到 `resNN` 命名（时间戳写入 `run_info.json`，并回填 `lr_type`）：
>
> ```bash
> python scripts/migrate_runs.py runs --default-lr-type unknown
> ```

## 断点续训

训练可以随时按 `Ctrl+C` 停止：每个 epoch 结束会写 `last.pth`，中断时也会保存一次当前进度（含 optimizer / scaler 状态）。之后用 `--resume` 接着**同一个 run** 训练，日志追加到同一个 `train_log.csv`，不会新建目录。

```bash
# 接着某个 run 继续（把 resNN 换成实际目录，例如 res05）
.venv/bin/python -m superresolution.train --resume runs/resNN --epochs 150

# 也可以省略 --epochs，沿用 checkpoint 里的总轮数
.venv/bin/python -m superresolution.train --resume runs/resNN
```

- `--resume` 可以给 run 目录（自动使用其中的 `last.pth`），也可以直接给 checkpoint 文件路径。
- 数据 / 模型 / 超参默认从 checkpoint 恢复。**续训不能切换 `--lr-type`、`--model`、`--scale` 等配置**（`--device` / `--num-workers` 除外）；要改配置请新开一个 run。
- `--epochs` 可以延长或缩短总轮数；改变总轮数会用新的 `T_max` 重新计算余弦调度，LR 曲线会随之变化。
- 学习率会按余弦调度接续到当前 epoch，不会跳回初始值。
- 若在 epoch 中途 `Ctrl+C`，会以当前 epoch 保存；续训从下一 epoch 开始，该 epoch 剩余的 batch 会被跳过（随机采样下影响可忽略）。
- 新 checkpoint 会恢复 optimizer / scaler 状态；旧 checkpoint（如 res04）没有 optimizer 状态，续训会重新初始化 Adam（会打印 warning），但 LR 仍正确衔接。

## 从已有 run 初始化

`--init-from` 只加载模型权重，不加载 optimizer / scheduler / epoch，适合用已有
run 做迁移学习或切换到新的 LR 类型：

```bash
.venv/bin/python -m superresolution.train \
  --model edsr --lr-type esp32 \
  --init-from runs/res05/best.pth \
  --lr 1e-4 --epochs 80 \
  --patch-size 192 --batch-size 16
```

- 可以给 checkpoint 文件，也可以给 run 目录；目录优先 `best.pth`，没有则用 `last.pth`
- checkpoint 的模型结构必须与当前 `--model` / `--channels` / `--n-blocks` 匹配
- `--resume` 与 `--init-from` 互斥
- 新 run 从 epoch 1 开始，`run_info.json` 会记录 `init_from` / `init_epoch` / `init_psnr`

## 推理

单张图片：

```bash
.venv/bin/python -m superresolution.infer \
  --checkpoint runs/res02/best.pth \
  --input lr.png --output sr.png
```

整个目录：

```bash
.venv/bin/python -m superresolution.infer \
  --checkpoint runs/res02/best.pth \
  --input lr_dir --output sr_dir
```

也可以使用安装的命令：`infer-sr --checkpoint ... --input ... --output ...`

| 参数 | 默认 | 说明 |
|---|---|---|
| `--tile` | `0` | 分块推理的块大小，`0` 为整图（大图省显存） |
| `--tile-overlap` | `8` | 分块重叠像素，减少拼接痕迹 |
| `--device` | `cuda` | 推理设备 |

输出命名：目录模式为 `<stem>_x{scale}.png`；单文件模式直接保存到 `--output` 指定的路径。

> checkpoint 会自描述模型结构（`model_name` / `model_kwargs`），任意 run 的 `best.pth` / `last.pth` 都可直接用于推理。

## 新增接口

- **生成 ESP32 LR 数据集**：

  ```bash
  python -m superresolution.degrade --version v1 --scale 4
  ```

  输出到 `data/DIV2K/DIV2K_train_LR_esp32/X4`、`DIV2K_valid_LR_esp32/X4`、
  `data/Flickr2K/Flickr2K_LR_esp32/X4`。支持 `--profiles`、`--severity`、
  `--splits`、`--output-format`、`--workers`、`--dry-run` 等参数。详见
  [src/superresolution/degrade/README.md](src/superresolution/degrade/README.md)。
- **ESP32 LR 训练**：生成后使用 `--lr-type esp32`，走原有对齐裁剪流程：

  ```bash
  python -m superresolution.train --model edsr --lr-type esp32 \
    --patch-size 192 --batch-size 16 --epochs 150
  ```
- **退化审查可视化**：`python -m servers.degrade_review_server --port 8002`，
  文档见 [servers/degrade_review_server/README.md](servers/degrade_review_server/README.md)。
- **超分推理可视化**：`python -m servers.sr_infer_server --port 8001`，
  文档见 [servers/sr_infer_server/README.md](servers/sr_infer_server/README.md)。

## 对齐说明

低分辨率图被视为高分辨率图按整数倍下采样。取块时严格对齐：

```
HR 起点 = (y·scale, x·scale)   HR 块 = (patch, patch)
LR 起点 = (y, x)               LR 块 = (patch/scale, patch/scale)
```

水平/垂直翻转与 90° 旋转增强同样同时作用于 HR/LR，保持像素对应关系，这是 PixelShuffle 类模型正确训练的前提。

## 项目结构

```
src/superresolution/
├── model.py      # ESPCN / EDSR 网络
├── dataset.py    # 对齐切割、bicubic/unknown 混合数据集
├── degrade/      # 版本化 ESP32 退化模型 + LR 数据集生成 CLI
├── train.py      # 训练 / 验证 / 断点续训
└── infer.py      # 推理
servers/          # 本地可视化服务（SR 推理 / 退化审查）
scripts/          # 数据集下载脚本 + runs 目录迁移脚本
```
