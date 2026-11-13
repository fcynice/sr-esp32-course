# 数据集与产物获取

本仓库**不包含** `data/` 下的图片和 `runs/` 下的权重：总数据约 30 GB，远超
GitHub 的合理仓库体积，也不符合 DIV2K / Flickr2K 的再分发约定。
本文说明每个数据集从哪来、解压成什么结构、如何校验，以及哪些小体积元数据
（manifest / summary）随仓库提交。

> 路径都相对于仓库根目录。`train.py` 顶部的 `TRAIN_SOURCES` / `VAL_SOURCES`
> 写死了下面这套目录名；换成自己的数据时保持同样的结构即可。

## 1. 总览

| 数据 | 内容 | 解压后大小 | 是否在仓库 |
|---|---|---:|---|
| DIV2K | 800 train HR + 100 valid HR + X2/X3/X4 bicubic/unknown LR | ~4.5 GB | 否 |
| Flickr2K | 2650 HR + X2/X3/X4 bicubic/unknown LR | ~22 GB | 否 |
| DIV2K + Flickr2K 压缩包 | 下载缓存 | ~3.1 GB | 否 |
| ESP32 v1 LR | 3550 张（1 个变体） | ~718 MB | 否 |
| ESP32 v2 LR | 7100 张（2 个变体） | ~1.5 GB | 否 |
| v1/v2 manifest + summary | 版本、指纹、数量、统计 | ~100 KB | **是**（`docs/manifests/`） |

训练默认同时读取 DIV2K 和 Flickr2K；只准备 DIV2K 会在 `Flickr2K/Flickr2K_HR`
处报 `FileNotFoundError`。如果只想快速跑通，先用 `scripts/smoke_test.py`
（不需要数据），或者只放少量样本并把 `Flickr2K` 目录补成同样的结构。

## 2. DIV2K

### 2.1 下载

```bash
# 方式 A：下载 DIV2K 全部（train/valid HR + 所有 LR），zip 落在 data/DIV2K/
bash scripts/download_div2k.sh

# 方式 B：只补 LR 归档，zip 落在 data/archives/DIV2K/
bash scripts/download_div2k_lr.sh
```

两个脚本有重叠：A 已经包含 LR，通常只跑 A 即可。脚本只负责下载，**不会自动解压**。

### 2.2 解压

```bash
cd data/DIV2K

# train HR（来自 hf-mirror 的 DIV2K_train.zip）
unzip -q DIV2K_train.zip

# valid HR + 各 LR 归档
for z in DIV2K_valid_HR.zip DIV2K_*_LR_*.zip; do
  unzip -q "$z"
done

# 如果用了方式 B，LR zip 在 data/archives/DIV2K/
cd ../../data/archives/DIV2K
for z in *.zip; do
  unzip -q "$z" -d ../../DIV2K
done
```

解压后应得到：

```text
data/DIV2K/
├── DIV2K_train_HR/                  # 800 张 .jpeg
├── DIV2K_valid_HR/                  # 100 张 .jpeg
├── DIV2K_train_LR_bicubic/X2|X3|X4/ # <stem>x{scale}.png
├── DIV2K_train_LR_unknown/X2|X3|X4/
├── DIV2K_valid_LR_bicubic/X2|X3|X4/
└── DIV2K_valid_LR_unknown/X2|X3|X4/
```

### 2.3 校验

```bash
find data/DIV2K/DIV2K_train_HR -maxdepth 1 -type f | wc -l   # 800
find data/DIV2K/DIV2K_valid_HR -maxdepth 1 -type f | wc -l   # 100
find data/DIV2K/DIV2K_train_LR_bicubic/X4 -maxdepth 1 -type f | wc -l  # 800
find data/DIV2K/DIV2K_valid_LR_unknown/X4 -maxdepth 1 -type f | wc -l  # 100
```

## 3. Flickr2K

Flickr2K 不在 DIV2K 官方包里，本仓库也没有附带下载脚本（公开源不稳定）。
请从 EDSR / NTIRE 的公开发布页获取，或直接向组长要现成数据包。

解压后需要整理成：

```text
data/Flickr2K/
├── Flickr2K_HR/                  # 2650 张 .png
├── Flickr2K_LR_bicubic/X2|X3|X4/ # <stem>x{scale}.png
└── Flickr2K_LR_unknown/X2|X3|X4/
```

LR 文件名必须是 `<HR stem>x{scale}.png`，例如 `000001.png` ↔ `000001x4.png`。
如果拿到的 LR 目录名不同，按上面的名字改名或建软链接即可。

校验：

```bash
find data/Flickr2K/Flickr2K_HR -maxdepth 1 -type f | wc -l  # 2650
find data/Flickr2K/Flickr2K_LR_bicubic/X4 -maxdepth 1 -type f | wc -l  # 2650
```

## 4. ESP32 退化 LR（由本仓库生成）

不下载现成数据也能生成：只要有 HR，退化模型是确定性的（相同版本 + seed）。

```bash
# 默认 v2，每张 HR 两个变体；输出约 1.5 GB
.venv/bin/python -m superresolution.degrade \
  --version v2 --variants 2 --scale 4 --seed 42

# v1 历史单变体
.venv/bin/python -m superresolution.degrade \
  --version v1 --variants 1 --scale 4 --seed 42
```

输出：

```text
data/DIV2K/DIV2K_train_LR_esp32_v2/X4/v0/  (v1/ 同理)
data/DIV2K/DIV2K_valid_LR_esp32_v2/X4/v0/  (v1/ 同理)
data/Flickr2K/Flickr2K_LR_esp32_v2/X4/v0/  (v1/ 同理)
# v1 只有 v0
```

数量与大小：

| 版本 | 变体 | 文件数 | 大小 |
|---|---|---:|---:|
| v1 | 1 | 3,550 | ~718 MB |
| v2 | 2 | 7,100 | ~1.5 GB |

`docs/manifests/v1/` 和 `docs/manifests/v2/` 里提交了 `manifest.json` 和 `summary.json`：

- `manifest.json`：版本、代码指纹、seed、severity、各 split 数量、输出目录；
- `summary.json`：按 profile / split 汇总的亮度、绿偏、饱和度、噪声等统计。

完整的 `manifest.jsonl`（逐文件 profile/seed/spec，v2 约 11 MB）未提交；只要
版本、seed、variants 相同，就可以重新生成同样的文件。核验代码指纹：

```bash
.venv/bin/python -c "from superresolution.degrade import fingerprint; print(fingerprint('v2'))"
# 4bee71cbf19db7a48ddc28232d096c311833a29962684187afbc0b78ceb4f28c
```

更多规则见 [`src/superresolution/degrade/README.md`](../src/superresolution/degrade/README.md)。

## 5. 权重与训练日志

- `.pth` 不提交（`res04` 起约 129 MB/个，超过 GitHub 100 MB 单文件限制）；
- 权重通过 GitHub Releases 分发，Release asset 是去掉 optimizer 的推理版
  （ESPCN 0.15 MB / EDSR 43 MB，共 8 个 run）：

  ```bash
  .venv/bin/python scripts/fetch_weights.py                 # 全部 8 个
  .venv/bin/python scripts/fetch_weights.py --only res05    # 只下 res05
  ```

  清单、SHA-256 和 PSNR 见 [`docs/RESULTS.md`](RESULTS.md)；
- 维护者可用 `scripts/export_weights.py` 从 `runs/resNN/best.pth` 导出 Release 文件；
- 每个 run 的 `args.json` / `run_info.json` / `train_log.csv` / `last_val.txt`
  已复制到 [`experiments/`](../experiments/)，用于复现实验配置。

## 6. 为什么不把数据放进 Git

- 30 GB 会让 clone 变得不可用；
- GitHub 单文件 100 MB 硬限制，LFS 免费额度也远小于 30 GB；
- DIV2K / Flickr2K 有各自的使用条款，仓库只应分发代码和生成脚本；
- ESP32 真实参考图 `data/esp32_camera/` 属于作者本地采集，不随仓库分发。
