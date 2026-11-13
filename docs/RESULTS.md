# 实验结果与权重

所有数字都在 **X4**、**整图验证**、RGB `[0,1]` 上测得。验证集与 `train.py`
完全一致：DIV2K valid 100 张 + Flickr2K 的 5% hold-out 132 张，共 **232 张**，
按 HR 图片划分，固定 `--seed 42`。`val_psnr` 取 `train_log.csv` 中的最大值。

## 1. 视觉效果

![SR comparison](images/sr_comparison.jpg)

左：bicubic ×4 放大；中：EDSR `res03`（bicubic LR 训练）；右：DIV2K HR。

![ESP32 degradation](images/esp32_degrade_v1_v2.jpg)

ESP32 退化对比（LR 用 nearest ×4 放大，避免额外插值掩盖噪声/模糊）：
v1 的残影/高光 bloom 更重，v2 削弱了 bloom/ghosting。

## 2. bicubic 基线 vs 模型

`bicubic` 基线 = 把 LR 用 bicubic 插值回 HR 尺寸后的 PSNR，使用与训练完全相同的
验证对。复现：

```bash
.venv/bin/python scripts/eval_baseline.py --lr-type bicubic
.venv/bin/python scripts/eval_baseline.py --lr-type unknown
.venv/bin/python scripts/eval_baseline.py --lr-type esp32_v1
.venv/bin/python scripts/eval_baseline.py --lr-type esp32_v2
```

| LR 类型 | bicubic 基线 | ESPCN | EDSR | EDSR 提升 |
|---|---:|---:|---:|---:|
| `bicubic` | 26.94 dB | 27.62（res02） | **28.61**（res03） | **+1.67 dB** |
| `unknown` | 21.94 dB | 22.90（res01） | **27.35**（res04） | **+5.41 dB** |
| `esp32_v1` | 15.79 dB | — | 19.46（res07，legacy） | +3.67 dB |
| `esp32_v2` | 15.90 dB | — | **19.56**（res08） | **+3.66 dB** |

> ESP32 的 PSNR 绝对值远低于 bicubic/unknown，是因为退化本身很重，不能跨 LR 类型
> 比较；有意义的是同一 LR 类型下相对 bicubic 基线的提升。

## 3. 每个 run 的配置

| run | 模型 | `--lr-type` | 完成/计划 epoch | patch/batch | 初始化 | best PSNR | 权重 |
|---|---|---|---:|---|---|---:|---|
| res01 | ESPCN | `unknown` | 42/100 | 128/16 | — | 22.90 | 152 KB |
| res02 | ESPCN | `bicubic` | 100/100 | 128/16 | — | 27.62 | 152 KB |
| res03 | EDSR 128/32 | `bicubic` | 150/150 | 192/16 | — | **28.61** | 43 MB |
| res04 | EDSR 128/32 | `unknown` | 150/150 | 192/16 | — | 27.35 | 129 MB |
| res05 | EDSR 128/32 | `both` | 118/150 | 192/16 | — | 28.07 | 129 MB |
| res06 | EDSR 128/32 | `esp32`（旧版） | 40/150 | 192/16 | — | 18.83 | 129 MB |
| res07 | EDSR 128/32 | `esp32`（旧版，实为 v1） | 80/80 | 192/16 | res05 | 19.46 | 129 MB |
| res08 | EDSR 128/32 | `esp32_v2` | 60/60 | 192/16 | res05 | **19.56** | 129 MB |

说明：

- `res05` 是 `both`（bicubic + unknown 混合），`28.07` 是两种 LR 合并后的平均值，
  不能和单类型数字直接横比。
- `res06` / `res07` 是旧代码产物：`run_info.json` 里仍是 `lr_type: "esp32"`，
  `res06` 的 `args.json` 还包含已删除的 `degrade_*` 参数。**用当前代码不能
  直接 `--resume` 或复现**，只作为历史记录保留。
- `res07` 实际读取的是 v1 数据；表中按 v1 展示。
- 权重不在 Git 仓库中。发布后在此补充 GitHub Releases / HuggingFace 链接：
  `TODO: release link`。

## 4. 复现命令

```bash
# EDSR + bicubic（res03）
.venv/bin/python -m superresolution.train \
  --model edsr --lr-type bicubic --scale 4 \
  --patch-size 192 --batch-size 16 --epochs 150

# EDSR + unknown（res04）
.venv/bin/python -m superresolution.train \
  --model edsr --lr-type unknown --scale 4 \
  --patch-size 192 --batch-size 16 --epochs 150

# EDSR + both（res05）
.venv/bin/python -m superresolution.train \
  --model edsr --lr-type both --scale 4 \
  --patch-size 192 --batch-size 16 --epochs 150

# 从 res05 初始化，适配 ESP32 v2（res08）
.venv/bin/python -m superresolution.train \
  --model edsr --lr-type esp32_v2 --scale 4 \
  --init-from runs/res05/best.pth --lr 1e-4 --epochs 60 \
  --patch-size 192 --batch-size 16
```

每个 run 的完整参数、日志和每张验证图的 PSNR 在
[`experiments/resNN/`](../experiments/)。

## 5. 指标口径

- PSNR 使用 `10 * log10(1 / MSE)`，输入范围 `[0,1]`，在整张验证图上计算；
- 推理结果先 `clamp(0,1)` 再算 PSNR；
- 验证阶段 `batch_size=1`，不做任何数据增强；
- 训练使用 L1 loss + Adam + CosineAnnealing，混合精度（CUDA）。

## 6. 发布权重（维护者）

`.pth` 不进 Git，推荐用 GitHub Releases（单文件上限 2 GB）：

```bash
gh release create x4-checkpoints \
  runs/res03/best.pth runs/res08/best.pth \
  --title "X4 checkpoints (ESPCN/EDSR)" \
  --notes "X4 checkpoints for bicubic / unknown / esp32_v2; see docs/RESULTS.md"
```

发布后把 asset 链接填到上面的 `TODO: release link`，并在 README「推理」一节保持
`runs/resNN/best.pth` 的路径约定（下载后放到对应 run 目录，或直接传 `.pth` 路径）。
