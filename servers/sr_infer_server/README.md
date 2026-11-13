# SR Inference Visualiser

一个只在本机运行的超分推理可视化服务：读取本地图片目录，选择 `runs/resNN` 下的
`best.pth` / `last.pth` 做推理，并在页面上对比推理前后结果。

设计上**不做文件上传**：服务端和图片在同一台机器上，页面通过目录浏览器选择图片，
服务端直接用该路径读取，避免复制和临时文件。

> **运行前提**：`runs/`、`*.pth` 和 `data/` 都不在 Git 仓库里。fresh clone 后服务
> 能启动，但 run 列表和默认目录可能为空。先下载权重：
>
> ```bash
> .venv/bin/python scripts/fetch_weights.py
> # 或只下主力
> .venv/bin/python scripts/fetch_weights.py --only res03,res05,res08
> ```
>
> 数据获取见 [`docs/DATASETS.md`](../../docs/DATASETS.md)。

## 功能

- 服务端目录浏览：面包屑、快捷入口、子目录、当前目录图片缩略图；**不递归**，只显示
  `.png` / `.jpg` / `.jpeg`。
- 模型选择：自动扫描 `runs/res*`，选择 run 后选择 `best.pth` 或 `last.pth`。
- 训练信息：展示 `run_info.json` / `args.json` / `train_log.csv` 中的架构、训练倍数、
  `bicubic` / `unknown` / `esp32_vN`、通道数、残差块数、patch、batch、epoch、学习率、最优验证 PSNR 等。
- 推理对比：
  - 左右并排模式：LR 与 SR 完整并排显示。
  - 拖动线模式：两图叠放，用一条可拖动、可调角度的细线分割；线一侧是推理前，
    另一侧是推理后；可拖动线本身或中间的圆点移动位置。
  - 滚轮缩放、拖动画布平移；左右两栏同步。
- HR 配对：DIV2K / Flickr2K 下的 LR 目录自动识别，不再维护固定清单。只要目录位于
  已知 split 下且包含 `X<scale>/<file>`，就会自动匹配同级 `<base>_HR`（推荐命名
  `<base>_LR_<variant>`）；此时有 LR、HR、SR 三张图，可切换 `LR / SR`、`SR / HR`、
  `LR / HR` 对比；LR 卡片/标签会显示来源类型（如 `esp32_v1/v0`）。
- 有 HR 且尺寸一致时，计算并显示 SR vs HR 的 PSNR / SSIM。
- 推理在后台任务中执行，页面轮询进度；大图自动分块，避免显存不足。
- SR 结果保存到 `data/sr_infer/results/`，支持下载。

## 目录结构

```
servers/sr_infer_server/
├── __init__.py
├── __main__.py          # python -m servers.sr_infer_server
├── config.py            # 路径、设备、分块、缩略图等配置
├── paths.py             # 允许目录校验 / 路径显示
├── datasets.py          # 自动发现 DIV2K/Flickr2K 的 LR -> HR 配对
├── browse.py            # 目录浏览、图片信息、缩略图缓存
├── runs.py              # 扫描 runs/resNN 与训练参数
├── metrics.py           # PSNR / SSIM（基于 torch，无额外依赖）
├── inference.py         # 模型缓存 + 推理 + 结果落盘
├── jobs.py              # 单 GPU 串行后台任务
├── main.py              # FastAPI app
├── static/              # 免构建前端（HTML/CSS/JS）
├── requirements.txt
└── README.md
```

## 启动

```bash
cd /path/to/superresolution

# 推荐：模块入口，默认 127.0.0.1:8001
uv run python -m servers.sr_infer_server

# 或
uv run uvicorn servers.sr_infer_server.main:app --host 127.0.0.1 --port 8001
```

打开 <http://127.0.0.1:8001/>，API 文档在 <http://127.0.0.1:8001/docs>。

启动后先自检：

```bash
curl http://127.0.0.1:8001/api/health
# 返回 GPU、run 数量、默认浏览目录等
```

常用参数：

```bash
uv run python -m servers.sr_infer_server \
  --host 127.0.0.1 --port 8001 \
  --runs-dir runs \
  --results-dir data/sr_infer/results \
  --device auto \
  --auto-tile 512 --tile-overlap 16 \
  --allowed-root data \
  --default-browse data \
  -v
```

## 配置

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `SR_INFER_HOST` | `127.0.0.1` | 监听地址 |
| `SR_INFER_PORT` | `8001` | 端口 |
| `SR_INFER_RUNS_DIR` | `runs` | 训练运行目录 |
| `SR_INFER_DATA_DIR` | `data` | 数据根目录 |
| `SR_INFER_RESULTS_DIR` | `data/sr_infer/results` | SR 结果目录 |
| `SR_INFER_ALLOWED_ROOTS` | 项目根 | 允许浏览的目录，`:` 分隔 |
| `SR_INFER_DEFAULT_BROWSE` | `data` | 页面默认打开的目录 |
| `SR_INFER_DEVICE` | `auto` | `auto` / `cpu` / `cuda` / `cuda:0` |
| `SR_INFER_TILE` | `512` | 大图分块大小 |
| `SR_INFER_TILE_THRESHOLD` | `512` | 超过该边长启用分块 |
| `SR_INFER_TILE_OVERLAP` | `16` | 分块重叠像素 |
| `SR_INFER_MAX_PIXELS` | `80000000` | 输入图像像素上限 |
| `SR_INFER_MAX_OUTPUT_PIXELS` | `100000000` | 输出图像像素上限（防止大图 OOM） |
| `SR_INFER_THUMB_SIZE` | `320` | 缩略图长边像素 |

允许浏览的根目录默认只有项目根；如需浏览其他图片目录，可用
`SR_INFER_ALLOWED_ROOTS=/path/a:/path/b`（Linux/macOS 用 `:`，Windows 用 `;`）
或重复传 `--allowed-root`。
出于安全考虑，接口会拒绝根目录之外的路径。

## 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/` | 可视化页面 |
| `GET` | `/api/health` | 健康检查、GPU、run 数量、默认目录 |
| `GET` | `/api/config` | 运行配置摘要 |
| `GET` | `/api/runs` | 所有 run、checkpoint、训练参数 |
| `GET` | `/api/browse?path=data` | 列出目录（不递归） |
| `GET` | `/api/image-info?path=...` | 图片尺寸、HR 配对信息 |
| `GET` | `/api/image?path=...&variant=thumb\|full` | 缩略图或原图 |
| `POST` | `/api/infer` | 提交推理任务 |
| `GET` | `/api/jobs/{job_id}` | 查询任务进度 / 结果 |
| `GET` | `/docs` | Swagger UI |

提交推理：

```bash
curl -X POST http://127.0.0.1:8001/api/infer \
  -H 'Content-Type: application/json' \
  -d '{
    "image_path": "data/DIV2K/DIV2K_valid_LR_bicubic/X4/0806x4.png",
    "checkpoint_path": "runs/res03/best.pth"
  }'
```

返回 `job_id`，然后轮询：

```bash
curl http://127.0.0.1:8001/api/jobs/<job_id>
```

## HR 配对规则

DIV2K / Flickr2K 的 LR -> HR 配对是**自动**的，不再维护固定目录清单。

规则：

1. 扫描根目录：`data/DIV2K`、`data/Flickr2K`；
2. 目录名推荐为 `<base>_LR_<variant>`（也支持 `<base>_LR`）；已知 split 下也兼容
   不带 `_LR` 的变体目录，如 `DIV2K_train_degraded_v3`；
3. 该目录下存在 `X<scale>/<file>` 路径；
4. 自动匹配同级 `<base>_HR` 目录（HR 体系保持固定）。

例如：

| LR 路径 | 自动匹配 HR |
|---|---|
| `data/DIV2K/DIV2K_train_LR_*/X*` | `data/DIV2K/DIV2K_train_HR` |
| `data/DIV2K/DIV2K_valid_LR_*/X*` | `data/DIV2K/DIV2K_valid_HR` |
| `data/Flickr2K/Flickr2K_LR_*/X*` | `data/Flickr2K/Flickr2K_HR` |
| `data/DIV2K/DIV2K_test_LR_*/X*`（未来） | `data/DIV2K/DIV2K_test_HR`（需存在） |

因此新增退化版本无需改代码，例如：

```text
data/DIV2K/DIV2K_train_LR_v2/X4/0001x4.png
data/DIV2K/DIV2K_train_LR_esp32_severity2/X4/0001x4.png
data/Flickr2K/Flickr2K_LR_profileA/X4/000001x4.png
```

也支持在 `X<scale>` 前后继续加子目录，此时页面的 `lr_type` 会带上这些子目录名：

```text
data/DIV2K/DIV2K_train_LR_esp32_v1/X4/v0/0001x4.png
  -> lr_type = esp32_v1/v0

data/DIV2K/DIV2K_valid_LR_esp32_v2/X4/v1/0801x4.png
  -> lr_type = esp32_v2/v1

data/DIV2K/DIV2K_train_LR_esp32/profileA/X4/0001x4.png
  -> lr_type = esp32/profileA
```

`X2` / `X3` / `X4` / `X8` 等任意 `X<scale>` 目录都会被识别，`X<scale>` 下面的
`v0`、`v1` 等 profile 子目录也不需要改代码。已知 split 下即使目录名不含 `_LR`
（例如 `DIV2K_train_degraded_v3`），只要包含 `X<scale>` 也会自动匹配。

文件名去掉 LR 末尾的 `xN` 后去 HR 目录找同名文件：

- DIV2K：`0075x4.png` -> `0075.jpeg`
- Flickr2K：`000969x4.png` -> `000969.png`
- ESP32：`0801x4.png` -> `0801.jpeg`（与 bicubic/unknown 使用同一份 HR）

其他目录（`data/大学教室`、`data/esp32_camera` 等）没有 HR，页面只显示 LR / SR。
新增 LR 版本时只要遵守上述目录命名和 `X<scale>` 结构即可自动生效。

## 推理与结果

- 模型和 checkpoint 结构直接复用 `src/superresolution/infer.py` 的 `load_model()` /
  `super_resolve()`，旧的 ESPCN checkpoint 仍然兼容。
- 模型按 `(checkpoint 路径, mtime, device)` 缓存，最多 4 个。
- 后台任务单线程串行执行，避免多个请求同时占用 GPU。
- 输入最大边超过 `SR_INFER_TILE_THRESHOLD` 时自动分块，默认 `tile=512`、
  `overlap=16`；否则整图推理。输出像素超过 `SR_INFER_MAX_OUTPUT_PIXELS` 时会直接报错，
  避免一次推理占满显存/内存。
- 结果保存为 `data/sr_infer/results/<时间戳>_<图片名>_x<scale>.png`。
- PSNR/SSIM 仅在 HR 与 SR 尺寸完全一致时计算；尺寸不一致时页面会显示提示。
- `data/` 已在 `.gitignore` 中忽略，结果不会进入版本库。

## 注意

- 服务默认只监听 `127.0.0.1`，适合本机使用；如需局域网访问请自行改 `--host`
  并评估目录浏览接口的暴露风险。
- 目录浏览接口只允许访问配置的允许根目录；不要把它直接暴露到公网。
- 图片格式仅支持 `.png` / `.jpg` / `.jpeg`。


## 常见问题

| 现象 | 原因 / 解决 |
|---|---|
| run 列表为空 | `runs/` 没有权重；先 `scripts/fetch_weights.py`，或自己训练一个 run |
| 页面目录为空 | `data/` 没下载，或 `--default-browse` 指向了不存在的目录 |
| `403` / 路径被拒绝 | 路径不在 `--allowed-root` 内；用 `SR_INFER_ALLOWED_ROOTS` 或重复 `--allowed-root` 添加 |
| CUDA OOM | 降低 `--auto-tile`（如 256），或先缩小输入图；也可以换更小的模型/权重 |
| 端口被占用 | `--port 8011` 换一个端口 |
| 想局域网访问 | `--host 0.0.0.0`，但目录浏览接口会暴露给同网段，注意安全 |
