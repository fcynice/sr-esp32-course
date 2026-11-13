# 实验记录（不含权重）

每个 `resNN/` 是从 `runs/resNN/` 复制出来的**小体积元数据**：

| 文件 | 内容 |
|---|---|
| `args.json` | 本次运行的完整 CLI 参数 |
| `run_info.json` | 序号、时间、模型、scale、`lr_type`、`model_kwargs` 等 |
| `train_log.csv` | 每轮 lr / train_l1 / val_psnr / 耗时 |
| `last_val.txt` | 最近一次验证每张图的 PSNR |

`best.pth` / `last.pth` **没有提交**（`res04` 起约 129 MB/个，超过 GitHub 单文件
100 MB 限制）。权重通过 GitHub Releases / HuggingFace 分发，结果汇总见
[`docs/RESULTS.md`](../docs/RESULTS.md)。

## Run 一览

| run | 模型 | LR 类型 | best PSNR | 备注 |
|---|---|---|---:|---|
| res01 | ESPCN | unknown | 22.90 | 42/100 epoch |
| res02 | ESPCN | bicubic | 27.62 | 100/100 epoch |
| res03 | EDSR 128/32 | bicubic | **28.61** | 150/150 epoch |
| res04 | EDSR 128/32 | unknown | 27.35 | 150/150 epoch |
| res05 | EDSR 128/32 | both | 28.07 | 118/150 epoch |
| res06 | EDSR 128/32 | esp32（旧版） | 18.83 | 40/150 epoch，legacy |
| res07 | EDSR 128/32 | esp32（旧版，v1） | 19.46 | 80/80 epoch，legacy |
| res08 | EDSR 128/32 | esp32_v2 | **19.56** | 60/60 epoch |

> `res06` / `res07` 是旧代码产物：`lr_type` 仍是 `esp32`，`res06` 还带已删除的
> `degrade_*` 参数，**当前代码不能直接 resume/复现**。保留它们只是为了记录历史。
> `res07` 实际读取 v1 数据。

## 使用方式

```bash
# 查看某次运行的参数
cat experiments/res08/args.json

# 画 PSNR 曲线
.venv/bin/python - <<'PY'
import csv
from pathlib import Path
for run in ("res03", "res04", "res08"):
    rows = [r for r in csv.DictReader(open(f"experiments/{run}/train_log.csv")) if r["val_psnr"]]
    print(run, [(r["epoch"], r["val_psnr"]) for r in rows[-3:]])
PY
```
