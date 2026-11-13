# Degrade Review Server

本地 ESP32 退化审查可视化服务。用于人工对比：

- 原始 HR（DIV2K / Flickr2K）
- 某个冻结退化版本（如 `v1`）生成的模拟 LR
- `data/esp32_camera/data01` 中的真实 ESP32 摄像头图片

服务只读取审查目录，不会修改图片，也不参与训练。当前训练读取由
`superresolution.degrade` 离线生成的 ESP32 LR 数据集。

> **运行前提**：真实参考图 `data/esp32_camera/data01` 和历史 review session
> `data/esp32_degrade_review/` 都在作者本地，**不随仓库分发**。fresh clone 后服务
> 可以启动，但 session / 参考图列表为空；这不影响训练和 LR 数据生成。

## 启动

```bash
cd /path/to/superresolution
uv run python -m servers.degrade_review_server --host 127.0.0.1 --port 8002
```

打开 <http://127.0.0.1:8002>。

可选参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--host` | `127.0.0.1` | 监听地址 |
| `--port` | `8002` | 端口，避开 8000 / 8001 |
| `--review-dir` | `data/esp32_degrade_review` | 审查 session 根目录 |
| `--reference-dir` | `data/esp32_camera/data01` | 真实 ESP32 参考图目录 |
| `--thumb-size` | `420` | 缩略图最长边 |
| `--reload` | — | uvicorn 热重载 |
| `-v` / `--verbose` | — | 打印详细日志 |

也可以用环境变量（适合 tmux / systemd）：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `DEGRADE_REVIEW_HOST` | `127.0.0.1` | 监听地址 |
| `DEGRADE_REVIEW_PORT` | `8002` | 端口 |
| `DEGRADE_REVIEW_ROOT` | `data/esp32_degrade_review` | 审查 session 根目录 |
| `DEGRADE_REVIEW_REFERENCE` | `data/esp32_camera/data01` | 真实 ESP32 参考图目录 |
| `DEGRADE_REVIEW_THUMB_SIZE` | `420` | 缩略图最长边 |

后台常驻：

```bash
nohup uv run python -m servers.degrade_review_server --host 127.0.0.1 --port 8002 \
  > runs/degrade_review_server.log 2>&1 &
```

## 审查图从哪里来

服务读取 `data/esp32_degrade_review/<session>/` 下的 `manifest.json` /
`manifest.jsonl`。当前 `superresolution.degrade` 已改为生成训练用 ESP32 LR
数据集（见 `src/superresolution/degrade/README.md`），不再生成新的 review
session。

已有的历史 session（例如 `v1_review_20261020/`）仍可直接查看。若后续需要新的
审查 session，按下面的格式新建一个目录即可，服务端代码无需改动。

### Session 目录结构

Session 的**目录名就是 `id`**（如 `v1_review_20261020`）。

```text
data/esp32_degrade_review/my_session/
├── manifest.json              # 必需：session 元信息
├── manifest.jsonl             # 每行一条审查记录（没有则显示空 session）
├── summary.json               # 可选：统计汇总
├── images/                    # 模拟 LR（路径相对于 session 目录）
│   └── 0001x4.png
└── previews/                  # 可选：contact_sheet.jpg / reference_sheet.jpg
```

### `manifest.json`

```json
{
  "created": "2026-11-13T12:00:00",
  "version": "v2",
  "fingerprint": "4bee71cbf19db7a48ddc28232d096c311833a29962684187afbc0b78ceb4f28c",
  "scale": 4,
  "severity": 1.0,
  "seed": 42,
  "record_count": 1,
  "profiles": ["green_indoor"],
  "source_counts": {"div2k/train": 1}
}
```

`record_count`、`profiles`、`source_counts` 主要用于列表页展示；`version` /
`fingerprint` 用来追溯生成配方。

### `manifest.jsonl`

每行一条 JSON。最小可用字段：

```json
{
  "id": "div2k_train_0001_green_indoor_123",
  "source_group": "div2k",
  "split": "train",
  "source": "data/DIV2K/DIV2K_train_HR/0001.jpeg",
  "lr": "images/0001x4.png",
  "profile": "green_indoor",
  "version": "v2",
  "scale": 4,
  "severity": 1.0,
  "seed": 123
}
```

- `lr`：**相对 session 目录**的模拟 LR 路径，必需；
- `source`：HR 路径，相对于项目根或绝对路径；不存在时只显示 LR/SR；
- `profile` / `version` / `scale` / `seed`：页面展示和筛选用；
- `spec` / `stats`：可选。`spec` 是完整退化参数，`stats` 是亮度/绿偏/噪声等指标，
  有的话详情页会展示；
- `source_group` / `split`：用于“来源”筛选和统计。

`summary.json` 和 `previews/*.jpg` 都是可选的；没有时页面仍能正常显示单条记录。

## 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 前端页面 |
| GET | `/api/health` | 健康检查 |
| GET | `/api/config` | 运行时配置 |
| GET | `/api/sessions` | session 列表 |
| GET | `/api/session/{id}` | session 元信息 + 全部审查记录 |
| GET | `/api/reference` | 真实 ESP32 参考图列表 |
| GET | `/api/image?path=...&variant=full\|thumb` | 图片 / 缩略图 |

`path` 使用项目根相对路径，或允许根目录内的绝对路径；服务会做沙箱校验。

## 前端功能

- 左侧选择 session、按 profile / 来源过滤、搜索文件名或 seed
- 网格缩略图浏览模拟 LR
- 点击卡片打开详情：
  - HR 原图与模拟 LR 并排
  - profile、seed、JPEG quality、绿偏、噪声、清晰度等指标
  - 完整退化参数 JSON
  - 底部真实 ESP32 参考图条
  - 键盘 `←` / `→` 切换，`Esc` 关闭
- “真实 ESP32 参考”页浏览 `data01` 全部实拍图

## 版本约定

`v1` 的 profile 范围和管线在人工审核后视为**冻结**。需要调整时新增
`src/superresolution/degrade/v2.py` 并在 `src/superresolution/degrade/__init__.py`
注册，不要直接改 `v1.py`。共享的 `ops.py` 也属于 `v1` 指纹的一部分；
后续版本如需新的底层算子，请新增独立模块，避免影响旧版本。旧 session 始终可以用旧版本复现。

## 依赖

见 `requirements.txt`，与项目根 `pyproject.toml` 保持一致。


## 常见问题

| 现象 | 原因 / 解决 |
|---|---|
| session 列表为空 | `--review-dir` 下没有 `manifest.json`；或数据没下载（仓库不包含 review session） |
| 点开 session 后没有图片 | `manifest.jsonl` 为空，或 `lr` 路径写错（必须相对 session 目录） |
| HR 显示“不可用” | `source` 指向的 HR 不在项目根/允许目录内，或文件不存在 |
| 参考图列表为空 | `--reference-dir` 下没有图片；默认 `data/esp32_camera/data01` 不随仓库提供 |
| 图片 403 | 路径不在允许范围；服务只允许项目根、`--review-dir`、`--reference-dir` |
| 端口被占用 | `--port 8012` 换一个端口 |
