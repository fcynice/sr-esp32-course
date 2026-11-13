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
审查 session，可以复用同一 manifest 格式单独生成；服务端代码无需改动。

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
