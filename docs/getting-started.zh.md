# 用 CPU 跑通第一次决策

[English walkthrough](getting-started.md) · [实际输出](../recipe/laya/validation.md#decision-demo) ·
[模型与硬件支持](supported-models.md)

System1-Omni 可以为 agent 提供结构化决策，例如把工单分给账单团队、
评估紧急程度、判断客户是否要求退款。这里用 **LAYA 英文检查点的 CPU worker**
跑通「安装 → worker → Rust frontend → 健康检查 → 退款判断」。
这条路径不需要 GPU、权重导出或 CUDA 编译，支持文本 `choice`、`score`、`noul`。
它使用上游 Python 模型执行，不验证原生 Rust 模型执行，也不支持本示例之外的
图片、音频或视频推理。

本页是[英文指南](getting-started.md)的简明中文入口。更新命令、依赖、能力范围或
链接时，两页应在同一 PR 中同步修改；完整排障和复现记录要求见英文版。
中文说明不代表该英文检查点已验证中文输入，请保留下面的英文请求。

## 开始前

- Linux x86_64 CPU，worker 使用 4 个 PyTorch 线程。Apple Silicon Mac 按文末说明
  修改两条安装命令；使用 Apple GPU 见 [MPS 指南](../recipe/laya/apple-silicon.md)。
- 为短文本示例预留 8 GB 主机内存、6 GB 空闲磁盘，用于环境、模型和 Rust 构建，
  不含工具链安装。这是规划余量，不是实测最低配置；长输入和额外模型需要更多资源。
- 安装 Git、curl、带 `venv`/pip 的 **Python 3.12**、Rust stable/Cargo、
  C 编译器和链接器。Debian/Ubuntu 的编译工具包为 `build-essential`。
  本次复现使用 Python 3.12.13、Rust/Cargo 1.98.1。
- 依赖固定为 `laya[serve]==0.3.20`、CPU 版 `torch==2.8.0+cpu`、
  `transformers==4.55.0` 等，完整直接依赖见
  [requirements-cpu.txt](../recipe/laya/requirements-cpu.txt)。MPS 环境使用另一份依赖文件。
- 首次启动从公开的 [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya)
  下载约 **846 MB** 的英文权重、分词器及编码器配置，无需 Hugging Face token。
  需要能访问 GitHub、PyPI、PyTorch CPU wheel 源和 Hugging Face。
- 准备三个终端，localhost **8000** 和 **8080** 端口应空闲。

记录中的缓存模型版本为 `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`；
新环境、空 Hub 缓存复现使用 `7b928d828b7b0e022f929d9bd2e44165aa270148`，两次均通过。
`laya-serve` 0.3.20 没有模型 revision 参数，默认下载 Hub 当前版本，后续下载的
概率值可能不同。安装、下载、编译、加载耗时应与推理延迟分别记录；没有固定的
安装时间承诺，本指南也不提供 CPU 延迟基准。

## 1. 安装和构建（终端 1）

已有仓库可从 `venv` 命令开始；已有 `.venv` 应先核对依赖。此后三个终端都在仓库根目录运行。

```sh
git clone https://github.com/ThinkFlowLab/system1-omni.git
cd system1-omni
python3.12 -m venv .venv
.venv/bin/python -m pip install 'torch==2.8.0+cpu' --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r recipe/laya/requirements-cpu.txt
cargo build -p omni-jev --release --locked
```

只构建 `omni-jev` frontend，不需要 CUDA 工具链。若 Python 缺少 `venv`/pip，
先安装对应组件；也可用 `uv venv --python 3.12 --seed .venv` 替换创建环境的命令。

## 2. 启动 worker（终端 1）

```sh
LAYA_HOST=127.0.0.1 LAYA_PORT=8000 LAYA_DEVICE=cpu \
LAYA_MODELS=english LAYA_PRELOAD=1 LAYA_THREADS=4 \
  .venv/bin/laya-serve
```

等待下载、加载完成，日志出现 `Uvicorn running on http://127.0.0.1:8000`。
保留进程运行；后续启动复用 Hugging Face 缓存。若失败，先查看此终端日志。

## 3. 检查 worker 并启动 frontend（终端 2）

```sh
curl --noproxy '*' --fail --silent --show-error http://127.0.0.1:8000/health
OMNI_JEV_BIND=127.0.0.1:8080 \
OMNI_JEV_BACKEND_URL=http://127.0.0.1:8000 \
  ./target/release/omni-jev
```

健康检查应返回 `{"status":"ok","loaded":["english"],"device":"cpu"}`。
frontend 应打印 `omni-jev listening on 127.0.0.1:8080` 和
`forwarding to http://127.0.0.1:8000/`，保留两个服务运行。
普通 `laya-serve` 的健康检查确认模型已加载，但不包含前向预热，首次请求仍可能较慢。

## 4. 发送退款判断（终端 3）

```sh
curl --noproxy '*' --fail --silent --show-error http://127.0.0.1:8080/health
curl --noproxy '*' --fail --silent --show-error http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"model":"english","state":"Please refund the duplicate charge.","questions":{"refund":{"type":"noul","instructions":"Does the customer ask for a refund?"}}}'
```

frontend 的健康响应应与 worker 一致。决策应返回 HTTP 200，
`answers.refund.type` 为 `noul`，`answers.refund.noul` 是模型对该问题的肯定概率值。
[实际运行](../recipe/laya/validation.md#decision-demo)中的答案为：

```json
{"refund":{"type":"noul","noul":0.8364,"confidence":0.8364,"answer_confidence":0.8364,"action":{"act_probability":1.0}}}
```

完整响应还有 `model`、`usage`、`routing`，其中 `routing.model` 为 `english`。
具体小数是本次运行的输出，不是准确率或校准效果保证。

## 5. 核对三类问题并记录环境（终端 3）

```sh
.venv/bin/python recipe/compare_with_backend.py --model english \
  --backend http://127.0.0.1:8000 --frontend http://127.0.0.1:8080
git rev-parse HEAD
.venv/bin/python -m pip freeze
.venv/bin/python - <<'PY'
from huggingface_hub import scan_cache_dir
for repo in scan_cache_dir().repos:
    if repo.repo_id == "convaiinnovations/laya":
        print("cached model revisions:", sorted(r.commit_hash for r in repo.revisions))
        for rev in repo.revisions:
            if "main" in rev.refs:
                print("cached main:", rev.commit_hash)
PY
```

预期五行 `PASS`：`health`、`department`、`urgency`、`refund`、`combined`，
均为 `status 200 -> 200`。脚本比较直连与 frontend 的状态码、内容类型和响应；
序列化或 usage 不同时也允许解析后的 `answers` 相等。这验证转发一致性，不测任务准确率。
在其他进程更新缓存前记录 `cached main`。

完成后分别在两个服务终端按 Ctrl-C。首次复现者可向
[issue #86](https://github.com/ThinkFlowLab/system1-omni/issues/86)提供 OS/CPU/RAM、
仓库 SHA、依赖版本、模型 revision、实际命令及首个失败或不清楚的步骤。
本次 agent 的已有环境复现不能替代独立首次使用者的报告。

端口占用时另选空闲端口，并同步修改环境变量、URL 和比较命令。
连接失败或 502 时先查 worker 日志和直连健康检查；504 表示 frontend 等待超时，
见[frontend 配置](../src/frontend/README.md)。下载失败或 `Fetching 5 files` 进度长时间不动时，
检查网络和缓存空间，按 Ctrl-C 后重新启动 worker；
只有完整模型已缓存时才能使用 `HF_HUB_OFFLINE=1`。启动时可能出现 `choice:11+`
温度被限制的警告，本示例及二选一检查仍可运行，受影响条目的置信度应视为未校准。

Apple Silicon Mac 没有 `torch==2.8.0+cpu` wheel，第 1 步的两条安装命令会报
`No matching distribution found for torch==2.8.0+cpu`。改用下面两条命令安装普通 macOS wheel
和其余固定依赖，其他步骤不变；编译器和链接器来自 Xcode Command Line Tools
（`xcode-select --install`）。在 M5 Pro、macOS 26.6 上，第 4、5 步的答案与实际运行记录一致。

```sh
.venv/bin/python -m pip install 'torch==2.8.0' --index-url https://download.pytorch.org/whl/cpu
sed 's/+cpu$//' recipe/laya/requirements-cpu.txt | .venv/bin/python -m pip install -r /dev/stdin
```

下一步：[Apple MPS](../recipe/laya/apple-silicon.md)、
[原生 Open-Jev CUDA](../recipe/open_jev/native.md)、[支持矩阵](supported-models.md)。
[Open-Jev H200 结果](../recipe/open_jev/validation.md)来自另一组单候选 GPU 工作负载，
不能当作本 CPU 示例的性能。
