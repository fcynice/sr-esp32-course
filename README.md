# Super-Resolution (ESPCN + PixelShuffle)

基于 PyTorch 的图像超分辨率项目。网络采用 ESPCN 风格结构：**三个卷积层 + 一个亚像素上采样层（`nn.PixelShuffle`）**，在 DIV2K 与 Flickr2K 数据集上训练。

## 网络结构

RGB 输入、放大倍数 `r`（默认 4）：

| 层 | 卷积核 | 通道 | 激活 |
|---|---|---|---|
| Conv1 | 5×5 | 3 → 64 | ReLU |
| Conv2 | 3×3 | 64 → 32 | ReLU |
| Conv3 | 3×3 | 32 → 3·r² | 无 |
| PixelShuffle(r) | — | 3·r² → 3，同时 H×r, W×r | — |

- 最后一层不接激活，输出 `3·r²` 个通道，由 `PixelShuffle` 重排为 `r×r` 的空间块，完成上采样。
- X4 时 48 通道 → 3 通道，参数量约 37K。
- 通道数可通过 `--mid1` / `--mid2` 调整（如 `--mid2 64` 得到对称的 64→64→48）。

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

下载脚本（可选，数据已就位可跳过）：

```bash
bash scripts/download_div2k.sh       # 下载 DIV2K
bash scripts/download_div2k_lr.sh    # 下载 DIV2K 的 LR 归档 (bicubic + unknown)
```

> 训练默认使用 **`LR_bicubic`（干净下采样，标准超分基准）**，收敛更快、PSNR 更高，便于验证流程。
> 也可用 `--lr-type unknown` 切换到 **真实退化** LR（含重度模糊），更贴近实际场景但难度高很多。
> LR 与 HR 严格整数倍对齐（HR = r × LR），dataset 切割时按同一空间位置对齐取块。

### bicubic vs unknown（实测参考）

X4、相同网络与数据划分下的验证 PSNR：

| LR 类型 | 训练量 | 验证 PSNR |
|---|---|---|
| `bicubic` | 3 epoch | **25.7 dB（仍在上升）** |
| `unknown` | 42 epoch | 22.9 dB |

`unknown` 的 LR 是重度模糊退化（其到 HR 的 MAE 约为 bicubic LR 的 2 倍），3 层小网络难以完全还原；
即使对比 bicubic 上采样基线，模型也仅领先约 +0.9 dB。建议先用 `bicubic` 跑通，再尝试 `unknown` 或更大的网络。

### 数据划分

| 用途 | 来源 | 数量 |
|---|---|---|
| 训练 | DIV2K train + Flickr2K 的 95% | 800 + 2517 |
| 验证 | DIV2K valid + Flickr2K 的 5% | 100 + 133 |

验证在**整图**上计算 PSNR（标准 SR 评价协议）。Flickr 的验证子集用固定随机种子（默认 42）划分，保证可复现。

## 安装

```bash
uv sync        # 或已就绪的 .venv
```

依赖：`torch`、`torchvision`、`tqdm`（GPU 版通过 `pytorch-cu130` 索引安装）。

## 训练

```bash
.venv/bin/python -m superresolution.train \
  --scale 4 --patch-size 128 --batch-size 16 --epochs 100
```

或使用安装的命令：

```bash
train-sr --scale 4 --epochs 100
```

主要参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--data-root` | `data` | 数据集根目录 |
| `--output-dir` | `runs` | 输出目录 |
| `--scale` | `4` | 放大倍数（2/3/4/8） |
| `--lr-type` | `bicubic` | LR 类型：`bicubic`（干净，推荐）或 `unknown`（真实退化） |
| `--patch-size` | `128` | HR 训练块大小（须能被 scale 整除，对应 LR 32×32） |
| `--batch-size` | `16` | 批大小 |
| `--epochs` | `100` | 训练轮数 |
| `--lr` | `1e-3` | 初始学习率（Adam + CosineAnnealing） |
| `--mid1` / `--mid2` | `64` / `32` | 中间层通道数 |
| `--val-every` | `5` | 每多少 epoch 验证一次 |
| `--num-workers` | `4` | DataLoader 线程数 |
| `--no-augment` | — | 关闭翻转/旋转增强 |
| `--no-amp` | — | 关闭混合精度 |

训练产物保存在 `runs/espcn_x{scale}_{时间戳}/`：

- `best.pth` — 验证 PSNR 最优权重
- `last.pth` — 最新权重
- `train_log.csv` — 每轮 loss / PSNR / 耗时
- `last_val.txt` — 最近一次验证每张图的 PSNR
- `args.json` — 本次运行的完整参数

`args.json` 会随 checkpoint 一起保存，推理时据此自动还原网络结构。

## 推理

单张图片：

```bash
.venv/bin/python -m superresolution.infer \
  --checkpoint runs/espcn_x4_xxx/best.pth \
  --input lr.png --output sr.png
```

整个目录：

```bash
.venv/bin/python -m superresolution.infer \
  --checkpoint runs/espcn_x4_xxx/best.pth \
  --input lr_dir --output sr_dir
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--tile` | `0` | 分块推理的块大小，`0` 为整图（大图省显存） |
| `--tile-overlap` | `8` | 分块重叠像素，减少拼接痕迹 |
| `--device` | `cuda` | 推理设备 |

输出保存为 `*_x{scale}.png`。

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
├── model.py      # ESPCN 网络
├── dataset.py    # 对齐切割的数据集 + 数据加载
├── train.py      # 训练 / 验证
└── infer.py      # 推理
scripts/          # 数据集下载脚本
```
