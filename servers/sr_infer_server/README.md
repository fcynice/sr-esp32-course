# SR Inference Visualiser

一个只在本机运行的超分推理可视化服务：读取本地图片目录，选择 `runs/resNN` 下的
`best.pth` / `last.pth` 做推理，并在页面上对比推理前后结果。

设计上**不做文件上传**：服务端和图片在同一台机器上，页面通过目录浏览器选择图片，
服务端直接用该路径读取，避免复制和临时文件。

## 功能

- 服务端目录浏览：面包屑、快捷入口、子目录、当前目录图片缩略图；**不递归**，只显示
  `.png` / `.jpg` / `.jpeg`。
- 模型选择：自动扫描 `runs/res*`，选择 run 后选择 `best.pth` 或 `last.pth`。
- 训练信息：展示 `run_info.json` / `args.json` / `train_log.csv` 中的架构、训练倍数、
  `bicubic` / `unknown`、通道数、残差块数、patch、batch、epoch、学习率、最优验证 PSNR 等。
- 推理对比：
  - 左右并排模式：LR 与 SR 完整并排显示。
  - 拖动线模式：两图叠放，用一条可拖动、可调角度的细线分割；线一侧是推理前，
    另一侧是推理后；可拖动线本身或中间的圆点移动位置。
  - 滚轮缩放、拖动画布平移；左右两栏同步。
- HR 配对：如果选中的 LR 位于 DIV2K / Flickr2K 的已知 LR 目录，自动找到对应 HR，
  此时有 LR、HR、SR 三张图，可切换 `LR / SR`、`SR / HR`、`LR / HR` 对比。
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
├── datasets.py          # 写死的 LR -> HR 数据集映射
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
cd /home/tomato/projects/python_project/superresolution

# 推荐：模块入口，默认 127.0.0.1:8001
uv run python -m servers.sr_infer_server

# 或
uv run uvicorn servers.sr_infer_server.main:app --host 127.0.0.1 --port 8001
```

打开 <http://127.0.0.1:8001/>，API 文档在 <http://127.0.0.1:8001/docs>。

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
`SR_INFER_ALLOWED_ROOTS=/path/a:/path/b` 或重复传 `--allowed-root`。
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

只对以下目录做 LR -> HR 配对（`X2` / `X3` / `X4`）：

| LR 目录 | HR 目录 |
|---|---|
| `data/DIV2K/DIV2K_train_LR_bicubic/X*` | `data/DIV2K/DIV2K_train_HR` |
| `data/DIV2K/DIV2K_train_LR_unknown/X*` | `data/DIV2K/DIV2K_train_HR` |
| `data/DIV2K/DIV2K_valid_LR_bicubic/X*` | `data/DIV2K/DIV2K_valid_HR` |
| `data/DIV2K/DIV2K_valid_LR_unknown/X*` | `data/DIV2K/DIV2K_valid_HR` |
| `data/Flickr2K/Flickr2K_LR_bicubic/X*` | `data/Flickr2K/Flickr2K_HR` |
| `data/Flickr2K/Flickr2K_LR_unknown/X*` | `data/Flickr2K/Flickr2K_HR` |

文件名去掉 LR 末尾的 `xN` 后去 HR 目录找同名文件：

- DIV2K：`0075x4.png` -> `0075.jpeg`
- Flickr2K：`000969x4.png` -> `000969.png`

其他目录（`data/大学教室`、`data/esp32_camera` 等）没有 HR，页面只显示 LR / SR。
后续新增数据集时，在 `datasets.py` 的 `DATASET_MAPPINGS` 里加一行即可。

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
