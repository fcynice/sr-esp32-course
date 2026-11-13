# ESP32 退化数据集 manifest

这里保存 `superresolution.degrade` 生成 v1/v2 LR 数据集时的**小体积元数据**，
方便队友和评审在不下载 1–2 GB LR 图片的情况下确认：

- 用的是哪个冻结版本、代码指纹是否一致；
- 用了多少 HR、生成多少 LR、有哪些变体；
- 每个 profile / split 的统计特征（亮度、绿偏、饱和度、噪声等）。

原始文件在 `data/esp32_degrade_dataset/<version>/`，**不随仓库提交**。完整逐文件
清单 `manifest.jsonl`（v2 约 11 MB）也未提交；只要版本、seed、variants 相同，
可以确定性重生成。

## 文件

| 文件 | 说明 |
|---|---|
| `v1/manifest.json` | v1 运行头：版本、指纹、seed、severity、数量、输出目录 |
| `v1/summary.json` | v1 按 profile / split 的统计汇总 |
| `v2/manifest.json` | v2 运行头 |
| `v2/summary.json` | v2 统计汇总 |

## `manifest.json` 关键字段

```json
{
  "created": "2026-10-26T10:47:06",
  "version": "v2",
  "fingerprint": "4bee71...f28c",
  "scale": 4,
  "severity": 1.0,
  "seed": 42,
  "profiles": ["day_soft", "green_indoor", "..."],
  "profile_mode": "auto",
  "output_format": "png",
  "variants": 2,
  "counts": {
    "div2k_train": {"total": 1600, "files": 800, "queued": 1600, "skipped": 0},
    "div2k_valid": {"total": 200, "files": 100, "queued": 200, "skipped": 0},
    "flickr2k":    {"total": 5300, "files": 2650, "queued": 5300, "skipped": 0}
  },
  "generated": 7100,
  "errors": 0,
  "outputs": {
    "div2k_train": {"root": "DIV2K/DIV2K_train_LR_esp32_v2/X4", "variants": ["v0", "v1"]},
    "div2k_valid": {"root": "DIV2K/DIV2K_valid_LR_esp32_v2/X4", "variants": ["v0", "v1"]},
    "flickr2k":    {"root": "Flickr2K/Flickr2K_LR_esp32_v2/X4", "variants": ["v0", "v1"]}
  }
}
```

`counts.total = files × variants`；`generated` 是实际写出的图片数。

## 核验指纹

```bash
.venv/bin/python -c "from superresolution.degrade import fingerprint; print(fingerprint('v2'))"
# 4bee71cbf19db7a48ddc28232d096c311833a29962684187afbc0b78ceb4f28c
.venv/bin/python -c "from superresolution.degrade import fingerprint; print(fingerprint('v1'))"
# 555a9e08e8de9a57cc49b50564ddc8211cc8bff72bf972318b4150190c05ee9a
```

如果指纹与 `manifest.json` 不一致，说明 `v1.py` / `v2.py` / `ops.py` 被改动过，
生成的数据不再可追溯。

## 重生成

```bash
# v2：7100 张（2 variants）
.venv/bin/python -m superresolution.degrade \
  --version v2 --variants 2 --scale 4 --seed 42

# v1：3550 张（1 variant）
.venv/bin/python -m superresolution.degrade \
  --version v1 --variants 1 --scale 4 --seed 42
```

生成后新的 `manifest.json` / `summary.json` 会写到
`data/esp32_degrade_dataset/<version>/`，可与这里的文件对比。
