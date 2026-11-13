# ESP32 Camera Degradation (versioned, offline)

基于真实 `data/esp32_camera/data01` 校准的 ESP32 摄像头退化模型。当前流程是
**离线生成带版本、带变体的 LR 数据集**，再交给原有配对训练管线使用。

> **范围说明**：真实参考图 `data/esp32_camera/`、采集服务（`servers/esp32_capture_server/`）
> 和 Android 中转 App 是作者本地任务，**不随仓库分发**。队友不需要 ESP32 硬件：
> 只要有 DIV2K/Flickr2K 的 HR，就能用冻结的 v1/v2 配方生成同样的 LR。
> 参考图仅用于作者校准和人工审查；模型审核后已冻结。

本文档面向“使用退化模型/生成数据集”的队友。完整数据获取见
[`docs/DATASETS.md`](../../../docs/DATASETS.md)，最终 PSNR 见
[`docs/RESULTS.md`](../../../docs/RESULTS.md)。

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

## 校准方法与参考数据

参考集是作者本地用 ESP32 摄像头采集的图片，主目录为 `data/esp32_camera/data01`，
另有少量其他场景目录作为低权重补充。参考集的典型特征是：

- 室内/弱光为主，整体欠曝、暗部噪声明显；
- 荧光灯下的绿偏、低饱和；
- 镜头软焦、轻微失焦/运动模糊；
- 高光裁切 + bloom；
- 低质量 JPEG 压缩。

校准流程是“人工审查 + 统计对比”：用 `servers/degrade_review_server` 并排对比真实图、
模拟 LR 和 HR，同时用 `stats.py` 统计亮度、绿偏、饱和度、噪声等指标，再调整 profile
参数。统计结果保存在 `summary.json`，生成参数保存在 `manifest.json`。

参考图 `data01`（160 张，绝大多数为 320×480）与 v2 两个弱光 profile 的统计均值：

| 指标 | 真实 `data01` | v2 `green_indoor` | v2 `lowlight_noise` |
|---|---:|---:|---:|
| 亮度 `y_mean` | 82.4 | 95.3 | 73.8 |
| 亮度标准差 `y_std` | 34.0 | 30.4 | 24.5 |
| 绿偏 `green_cast` | 4.76 | 6.77 | 3.69 |
| 饱和度 `saturation` | 9.63 | 14.32 | 11.82 |

两个 profile 的亮度和绿偏大致把真实参考夹在中间；`day_soft` 等 profile 权重较低，
负责提供更干净/更亮的样本。参考图与模拟 LR 分辨率不同，`noise_sigma`、
`laplacian_var` 对分辨率敏感，因此上表只列尺度相对稳定的指标。真实列由
`data01` 逐图调用 `superresolution.degrade.stats.image_stats` 后取均值得到
（参考图不在仓库，队友无法直接复算）。

**参考图不随仓库分发**（体积 + 隐私），所以队友无法重跑校准；但 v1/v2 已冻结，
生成过程是确定性的，评审可以直接核对指纹和 manifest。

### v1 → v2 的主要变化

| 问题 | v1 | v2 |
|---|---|---|
| 高光周围黑圈 | bloom 较强（`0.15–0.55`） | bloom 降到 `0–0.08` |
| 残影/拖影 | ghosting 强度/位移较大 | 强度、位移、拖影均减弱 |
| 高光 | 容易过曝成块 | 用更柔和的 highlight knee |

> 结论来自人工审查：v1 训练时模型会尝试“反演” bloom，导致高光周围出现黑圈；
v2 削弱 bloom 后该现象消失。

## 退化管线

```text
1. resize 到 LR（bicubic/lanczos/bilinear）
2. 色差 chromatic_aberration
3. 高斯软焦 / 失焦 / 运动模糊
4. 残影 ghosting / 拖影 smear / 条带抖动 band_jitter
5. 色调、白平衡、高光 knee、bloom、banding、暗角
6. 传感器噪声（散粒/读出/色度/行列固定模式/热像素）
7. clip → posterize（可选）→ JPEG 编解码
```

顺序很重要：先做几何/镜头退化，再做色彩/噪声，最后才做 JPEG 压缩。
每个步骤对应 `ops.py` 里的同名函数；profile 参数在 `v1.py` / `v2.py` 的 `PROFILES` 中。

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

当前生成结果：

| 版本 | 变体 | 文件数 | 大小 |
|---|---|---:|---:|
| v1 | 1（`v0`） | 3,550 | ~718 MB |
| v2 | 2（`v0`, `v1`） | 7,100 | ~1.5 GB |

`manifest.json` 记录版本、指纹、seed、severity、各 split 数量；`summary.json`
记录按 profile/split 的统计；逐文件 `manifest.jsonl` 记录每张图的 profile、seed、
完整 spec。仓库只提交前两个（约 100 KB），完整 jsonl 可由相同 seed 重生成。
小体积副本在 [`docs/manifests/`](../../../docs/manifests/)，字段说明和指纹核验见
[`docs/manifests/README.md`](../../../docs/manifests/README.md)。

训练时 `--lr-type esp32_v2` 会读取 `LR_esp32_v2/X4`，并把 `v0`、`v1` 都识别为
同一 HR 的候选变体，每次取 patch 时随机选一个。验证固定使用 `v0`，保证不同
epoch 的 PSNR 可比。

> LR 图片本身不进 Git；队友要么运行生成命令，要么从 Release/网盘获取。
> 生成耗时取决于 CPU、磁盘和 workers 数量；v2 的 7,100 张建议预留几十分钟。

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

其它 introspection：

```bash
python -m superresolution.degrade --list-versions   # v1 v2
python -m superresolution.degrade --list-profiles --version v2
```

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

实测（X4，232 张验证图）：

| LR 类型 | bicubic 基线 | EDSR | 提升 |
|---|---:|---:|---:|
| `esp32_v1` | 15.79 dB | 19.46 dB（res07） | +3.67 dB |
| `esp32_v2` | 15.90 dB | **19.56 dB**（res08） | +3.66 dB |

完整结果见 [`docs/RESULTS.md`](../../../docs/RESULTS.md)。
- 历史的人工审查可视化服务仍在 `servers/degrade_review_server/`；当前生成器不再
  产生新的 review session。该服务依赖本地的 `data/esp32_camera/data01` 和
  `data/esp32_degrade_review/`，fresh clone 后列表为空是正常的。
