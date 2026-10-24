# Servers

本目录存放项目用到的所有独立服务（server）。每个服务**一个独立子文件夹**，代码、依赖说明、
启动方式、文档都自包含，方便后续继续新增互不干扰。

## 结构约定

```
servers/
├── README.md                    # 本文件：服务总览与新增规范
├── __init__.py                  # 使 servers 成为包（python -m servers.xxx 可用）
└── <server_name>/               # 一个服务 = 一个文件夹
    ├── __init__.py
    ├── __main__.py              # python -m servers.<server_name> 入口
    ├── config.py                # 环境变量 / CLI 配置
    ├── main.py                  # FastAPI app（或服务主逻辑）
    ├── requirements.txt         # 该服务单独所需依赖（便于移植）
    └── README.md                # 该服务自己的使用文档
```

命名全部用小写 + 下划线（如 `esp32_capture_server`），与目录名保持一致。

## 服务列表

| 服务 | 端口(默认) | 说明 | 文档 |
|---|---|---|---|
| [`esp32_capture_server`](./esp32_capture_server/) | `8000` | 接收 ESP32 摄像头 JPEG（经手机中转：Android App 或 Termux），保存到 `data/esp32_camera/` | [README](./esp32_capture_server/README.md) |
| [`sr_infer_server`](./sr_infer_server/) | `8001` | 超分推理可视化：本地目录选图、选择 `runs/resNN` 的 best/last 权重、LR/SR/HR 并排与可旋转拖动线对比（支持 bicubic / unknown / esp32 LR） | [README](./sr_infer_server/README.md) |
| [`degrade_review_server`](./degrade_review_server/) | `8002` | ESP32 退化审查：按 profile 浏览模拟 LR、HR 对照、退化参数与真实 `data01` 参考图 | [README](./degrade_review_server/README.md) |

端口分配时请避开已用端口，新增服务建议顺延（如 `8001`、`8002`）并在此表登记。

## 环境与依赖

统一使用项目根目录的 `uv` 虚拟环境，共享 `pyproject.toml`。

```bash
cd /home/tomato/projects/python_project/superresolution
uv sync                 # 安装/同步全部依赖
```

新增依赖优先用 `uv add <pkg>` 写入 `pyproject.toml`；`<server_name>/requirements.txt`
只作为「单独复制该服务到别的环境」时的参考清单，需与 `pyproject.toml` 保持同步。

## 通用启动方式

每个服务都提供 `python -m` 入口：

```bash
uv run python -m servers.<server_name> [参数]
```

也可以直接用 ASGI 服务器指定 app：

```bash
uv run uvicorn servers.<server_name>.main:app --host 0.0.0.0 --port <port>
```

后台常驻（本机 tmux / nohup）：

```bash
uv run nohup python -m servers.esp32_capture_server --port 8000 \
  > runs/esp32_capture_server.log 2>&1 &
```

## 新增一个 server 的步骤

1. `mkdir servers/<server_name>`，从已有服务复制骨架（`__init__.py`、`__main__.py`、
   `config.py`、`main.py`、`README.md`）。
2. 在 `config.py` 用 `dataclass` + 环境变量集中管理配置，`__main__.py` 里加 CLI 参数覆盖。
3. 在 `main.py` 暴露一个模块级 `app = FastAPI(...)`，路由统一加 `/api` 前缀。
4. 写清 `<server_name>/README.md`（用途、拓扑图、启动、接口表、示例）。
5. 回到本文件「服务列表」登记服务名、端口、说明、文档链接。
6. 如引入新依赖，`uv add` 后同步更新该服务的 `requirements.txt`。

## 通用约定

- **配置**：不硬编码路径/密钥，走环境变量 + CLI；密钥类不写进仓库。
- **数据落盘**：一律写到项目根的 `data/` 下（已在 `.gitignore` 忽略），并按服务名单独建子目录。
- **接口**：REST 风格，路径前缀 `/api`，返回 JSON；提供 `GET /api/health` 健康检查。
- **鉴权**：如对外暴露，使用共享 token（`X-Auth-Token` 头）等方式，默认关闭、按需开启。
- **日志**：用标准 `logging`，不 `print`；启动时打印监听地址与数据目录。
- **不修改**仓库外的系统文件；只读取需要的输入。

## 相关文档

- 项目总说明：[../README.md](../README.md)
- ESP32 采集链路：[./esp32_capture_server/README.md](./esp32_capture_server/README.md)
