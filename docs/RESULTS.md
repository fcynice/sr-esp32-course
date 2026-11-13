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
- 权重不在 Git 仓库中。Release asset 是去掉 optimizer/scheduler 的推理版：
  EDSR 约 **43 MB/个**，ESPCN 约 **0.15 MB/个**。

  | 文件 | run | 大小 | SHA-256（前缀） |
  |---|---|---:|---|
  | [`espcn_x4_unknown.pth`](https://github.com/fcynice/sr-esp32-course/releases/download/weights-x4-v1/espcn_x4_unknown.pth) | res01 | 0.15 MB | `3b855cda…` |
  | [`espcn_x4_bicubic.pth`](https://github.com/fcynice/sr-esp32-course/releases/download/weights-x4-v1/espcn_x4_bicubic.pth) | res02 | 0.15 MB | `bc85c05c…` |
  | [`edsr_x4_bicubic.pth`](https://github.com/fcynice/sr-esp32-course/releases/download/weights-x4-v1/edsr_x4_bicubic.pth) | res03 | 43.2 MB | `2fa46dc5…` |
  | [`edsr_x4_unknown.pth`](https://github.com/fcynice/sr-esp32-course/releases/download/weights-x4-v1/edsr_x4_unknown.pth) | res04 | 43.2 MB | `2c6dcdad…` |
  | [`edsr_x4_both.pth`](https://github.com/fcynice/sr-esp32-course/releases/download/weights-x4-v1/edsr_x4_both.pth) | res05 | 43.2 MB | `f34fbfcb…` |
  | [`edsr_x4_esp32_legacy_res06.pth`](https://github.com/fcynice/sr-esp32-course/releases/download/weights-x4-v1/edsr_x4_esp32_legacy_res06.pth) | res06 | 43.2 MB | `2e85e6db…` |
  | [`edsr_x4_esp32_v1_legacy_res07.pth`](https://github.com/fcynice/sr-esp32-course/releases/download/weights-x4-v1/edsr_x4_esp32_v1_legacy_res07.pth) | res07 | 43.2 MB | `f8fc86bc…` |
  | [`edsr_x4_esp32_v2.pth`](https://github.com/fcynice/sr-esp32-course/releases/download/weights-x4-v1/edsr_x4_esp32_v2.pth) | res08 | 43.2 MB | `a5724f75…` |

  > **res05 是后续真实退化训练的初始化基础**：`--init-from runs/res05/best.pth`
  > 只加载模型权重，Release 里的 43 MB 推理版就够用。

  下载到 `runs/resNN/best.pth`（脚本会校验 SHA-256）：

  ```bash
  # 全部 8 个（约 258 MB）
  .venv/bin/python scripts/fetch_weights.py

  # 只下载主力：res03 / res05 / res08
  .venv/bin/python scripts/fetch_weights.py --only res03,res05,res08

  # 本机已有同名 checkpoint 时默认跳过；只想验证 Release 可下到临时目录
  .venv/bin/python scripts/fetch_weights.py --dest /tmp/released_weights
  ```

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

`.pth` 不进 Git，推荐用 GitHub Releases（单文件上限 2 GB）。先导出推理版（去掉
optimizer，129 MB → 43 MB；8 个 run 全部导出）：

```bash
.venv/bin/python scripts/export_weights.py
# 输出 release_assets/*.pth（espcn 0.15 MB / edsr 43 MB）
```

然后创建 Release（tag 必须是 `weights-x4-v1`，否则下载链接会变）：

```bash
gh release create weights-x4-v1 \
  release_assets/*.pth \
  --target master \
  --title "X4 checkpoints (ESPCN / EDSR)" \
  --notes-file /tmp/sr-release/RELEASE_DESCRIPTION.md
```

没有 `gh` 就用网页：Releases → Create a new release → tag `weights-x4-v1` →
把 `release_assets/*.pth` 全部拖进去，正文粘贴
`/tmp/sr-release/RELEASE_DESCRIPTION.md` → Publish。发布后验证：

```bash
.venv/bin/python scripts/fetch_weights.py --dest /tmp/released_weights
```
