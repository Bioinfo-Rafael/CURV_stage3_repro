# CURV Stage 3 — 单步 GRPO 最小复现

本 repository **用 synthetic 图像和 tiny Qwen2.5-VL 执行生成 → reward → GRPO loss → backward → optimizer 更新**。复用官方 CURV prompt 和 format reward，通过 hard assertions 验证训练路径确实可运行。

真实图像 Stage 1/2 由 [CXR_LLM](../CXR_LLM/README.md)负责，图像分类由 [CXR_GRN](../CXR_GRN/README.md)负责。这里的成功不代表真实 MIMIC 训练或医学有效性验证。

[日本語](README.ja.md) | [English](README.en.md) | [简体中文](README.zh-CN.md)

[功能](#capabilities) · [目录](#layout) · [运行](#run) · [实现差异](#implementation) · [原文与历史](#archive)

<a id="capabilities"></a>

## 1. 验证哪些内容

| 步骤 | Smoke 检查 |
|---|---|
| 输入／生成 | 一个 synthetic 图像 prompt 生成两个 response，包含 VLM pixel 输入 |
| Reward | 官方 format＋明确标记的 smoke-only index reward，有限值、group std > 0 |
| GRPO | advantage、policy/reference logprob、KL、clipped objective 有限 |
| 训练 | 非零 gradient、backward、optimizer step、parameter changed |
| 官方代码 | 指定的 clean CURV checkout 在运行前后保持不变 |

在 CPU 上执行一步，更新整个 tiny model。这与 CXR_LLM 只更新一个 norm parameter 的 smoke 不同。

<a id="layout"></a>

## 2. Directory 结构

```text
CURV_stage3_repro/
├── scripts/
│   ├── run_stage3_smoke.sh        # Python environment launcher
│   ├── run_stage3_smoke.py        # GRPO step + hard assertions
│   └── run_stage3_local_smoke.py  # Preserve/restore untracked Finder metadata
├── configs/smoke_config.json     # Tiny model, generation, optimizer settings
├── mock_data/                    # Synthetic image + prompt
├── requirements/                 # Pinned Python dependencies
├── outputs/stage3_smoke_result.json
├── .venv/                        # Local environment (not committed)
└── audit.md                      # Fidelity/dependency audit
```

<a id="run"></a>

## 3. 运行方法

从本 repository root 执行。已有环境可以直接复用。

### 首次配置

验证版本：Python 3.11、torch 2.7.1、Transformers 5.14.1、TRL 1.9.2。仅在 `.venv` 不存在时创建。

```bash
conda create -p .venv python=3.11 pip -y
.venv/bin/python -m pip install -r requirements/requirements.txt
```

首次下载 tiny model／processor 需要网络；首次运行可移除下方 offline 变量，缓存后直接复用。不需要下载 MIMIC。

### Synthetic GRPO Smoke

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 bash scripts/run_stage3_smoke.sh \
  --curv-root ../CXR_LLM/models/CURV
```

`--curv-root` 指定官方 prompt／reward 所在 checkout。由于相邻 `CURV/README.md` 已在本地编辑，当前使用相同 commit 的未修改副本 `CXR_LLM/models/CURV`。这条命令不会复制模型或数据。

其他机器没有该副本时，请准备 clean 的官方 commit `f8bf7d0ad5f3c26e9336f118c78b6264887c947b` 并替换路径。不要为通过检查而 reset 或临时隐藏 tracked 修改，应指定另一份 clean checkout。只有未跟踪 root `.DS_Store` 是唯一障碍时，才用已有 `run_stage3_local_smoke.py --curv-root <path>` 临时保管并恢复 Finder metadata。

### 成功条件与输出

要求退出 code 0，并通过全部断言，包括 `GENERATION OK`、`REWARD STD > 0`、`GRPO LOSS FINITE`、`BACKWARD OK`、`OPTIMIZER STEP OK`、`PARAMETER CHANGED`、`UPSTREAM UNMODIFIED`。

结果写入 [outputs/stage3_smoke_result.json](outputs/stage3_smoke_result.json)。synthetic 图像不存在时以确定性方式生成。policy loss=0 不一定意味着 gradient=0，因此单独验证 policy-only 和 total-loss gradient。

<a id="implementation"></a>

## 4. 实现差异与范围限制

| 组件 | 官方 recipe／完整实验 | 本 smoke |
|---|---|---|
| Model／输入 | Stage 2 checkpoint＋真实数据 | Tiny Qwen2.5-VL＋一个 synthetic prompt |
| 优化实现 | ms-swift GRPO recipe | TRL 1.9.2＋观测 instrumentation |
| Reward | Format＋CheXbert／RadGraph 等 | 官方 format＋smoke-only variance reward |
| 生成数／长度 | 8 次，最长 1,536 tokens | 2 次，最长 16 tokens |
| 更新 | 分布式完整 schedule，学习率 `1e-7` | CPU 一步，`1e-4` |
| 官方 source | 参考实现 | 不修改，保留严格 clean／unchanged 断言 |

记录缺失 `lm_head.weight` 的初始化 warning。model 标记为 `MODEL MOCK / SMALL SUBSTITUTE`，辅助 reward 标记为 `SMOKE-ONLY REWARD; NOT PART OF CURV`。医学 reward 的依赖／checkpoint、真实图像 messages 和正式 Stage 2 model 齐备前，不应声称完整复现。

[审计记录](audit.md)与末尾表格保留了 library 尝试、替代范围和实验配置。全部 repository 的流程见已有[统一 RUN_GUIDE](../CXR_LLM/docs/RUN_GUIDE.zh-CN.md#stage3)。

<a id="archive"></a>

## 附录：既有原文、全部断言、配置比较与迁移历史

以下原样保留整理前的完整 README。绝对路径和旧默认启动示例属于历史记录；当前请使用上方明确指定 checkout 的运行方式。

<details>
<summary>整理前のREADME全文 / Previous README (verbatim) / 原README全文</summary>

# CURV Stage 3 最小 GRPO 复现

[日本語](README.ja.md) | [English](README.en.md) | **中文** | [语言选择](README.md)

## 快速运行

四个 repository 已移至 `/Users/cls-lab/Git/LinGu/`。
真实图像的 Stage 1/2 和 CLARITY smoke 由相邻的 CXR_LLM 执行，
NIH/MIMIC 真实图像 MRGL 由 CXR_GRN 执行。本 Stage 3 仍使用 synthetic 图像
验证 GRPO，不会自动使用手动 MIMIC 图像或 Stage 1/2 教师 JSON。
移动后的 `.venv` 可继续使用，无需重建环境或重新下载模型。
完整流程见已有的[统一 RUN_GUIDE](../CXR_LLM/docs/RUN_GUIDE.zh-CN.md)。

```bash
cd /Users/cls-lab/Git/LinGu/CURV_stage3_repro && bash scripts/run_stage3_smoke.sh
```

首次运行时，请先按照下方的[环境配置与运行](#环境配置与运行)创建`.venv`。

上述命令要求官方 CURV checkout 干净。若仅未跟踪的 root `.DS_Store` 阻碍检查，
在同一目录运行 `.venv/bin/python scripts/run_stage3_local_smoke.py`。
该 wrapper 临时保管并恢复 Finder metadata，不修改图像、upstream tracked 文件或既有断言。

本项目使用mock data实际执行一次完整的 **CURV Stage 3 GRPO control flow**，
同时保证不修改相邻官方`CURV/` checkout中的任何tracked file。

## 已复现的内容

- 官方CURV CXR system prompt -> synthetic image + user prompt
- 使用tiny Qwen2.5-VL architecture生成2个sampled responses
- 官方CURV `FormatRewardCXR`
- group内reward的mean/std与归一化advantage
- policy与固定reference的completion-token log probability
- TRL的clipped GRPO objective与有限KL
- backward、非零gradient、optimizer step以及parameter变化的实测验证

## 未复现的内容

- 论文性能或论文报告的数值
- 完整dataset以及medical validity
- 真实Stage 2 checkpoint及其训练过程
- CheXbert/RadGraph reward及其外部checkpoint
- 完整的8-GPU/vLLM/DeepSpeed ZeRO-3训练
- 1536-token report以及完整training schedule

模型始终标记为 **MODEL MOCK / SMALL SUBSTITUTE**。用于保证reward variance的
index-based reward始终标记为 **SMOKE-ONLY REWARD; NOT PART OF CURV**。

## 与官方CURV的对应关系

| 复现侧 | 官方CURV侧 | 作用 | 保真度 |
|---|---|---|---|
| `mock_cxr.jsonl` + synthetic PNG | original GRPO messages JSONL | training input | mock |
| runtime加载的prompt | `training/prompts/prompt_cxr.txt` | system prompt | 官方、verbatim |
| runtime加载的`FormatRewardCXR` | `training/reward_functions/format_reward.py` | format reward | 官方、直接执行 |
| TRL 1.9.2 `GRPOTrainer` | 保存的ms-swift GRPO recipe | optimization | 正式的等价外部实现 |
| tiny Qwen2.5-VL | Stage 2 checkpoint path | policy/reference VLM | small substitute |
| `smoke_only_index_reward` | 无 | 保证reward variance | smoke-only substitute |
| audit instrumentation subclass | 无 | 保存中间证据 | 仅用于观测 |

## Original配置与smoke配置

| 配置 | CURV original recipe | Smoke run |
|---|---:|---:|
| model | Stage 2 Qwen2.5-VL checkpoint | tiny Qwen2.5-VL test model（本环境加载6,288,704 parameters） |
| prompts | full dataset | 1个mock prompt |
| generations/prompt | 8 | 2 |
| max completion length | 1536 | 16 |
| temperature / top-p / top-k | 1.0 / 0.9 / 50 | 相同 |
| train type | full | tiny model的full update |
| learning rate | `1e-7` | `1e-4`，以便在一次float32 step中测量变化 |
| rewards | format + CheXbert + RadGraph variants | 官方format + 明确标记的smoke-only variance reward |
| accelerator | 8 GPUs、vLLM、ZeRO-3的设计意图 | CPU（MPS backward不兼容） |
| optimizer steps | full epoch | 1 |

## 环境配置与运行

已验证环境为Apple Silicon上的Python 3.11：

```bash
conda create -p .venv python=3.11 pip -y
.venv/bin/python -m pip install -r requirements/requirements.txt
```

在本目录中使用以下单条命令运行完整的one-step smoke test：

```bash
bash scripts/run_stage3_smoke.sh
```

如果synthetic image不存在，script会创建deterministic的224×224图像。实测结果写入
`outputs/stage3_smoke_result.json`。在tiny model与processor完成cache之前，
需要访问Hugging Face网络。

## Hard assertions

以下任意一项不满足时，命令会以non-zero状态退出：

```text
DATA LOAD OK
MODEL LOAD OK
PROCESSOR OK
GENERATION OK
NUM GENERATIONS >= 2
VLM PIXEL INPUT OK
CURV FORMAT REWARD OK
REWARD VECTOR FINITE
REWARD STD > 0
ADVANTAGES FINITE
POLICY LOGPROBS FINITE
KL FINITE
GRPO LOSS FINITE
POLICY-ONLY NONZERO GRADIENT
BACKWARD OK
NONZERO GRADIENT EXISTS
OPTIMIZER STEP OK
PARAMETER CHANGED
UPSTREAM UNMODIFIED
```

关于upstream code审计、placeholder、interface不一致、library path尝试以及复用决策，
请参阅`audit.md`。

当两个归一化advantage对称时，第一次step的clipped policy loss数值可能恰好为
`0.0`。但两个不同sampled token sequence具有不同的log-probability derivative，
因此这并不意味着gradient为零。script会分别assert policy-only gradient与实际
total-loss backward产生的gradient。

## 迁移到真实CURV Stage 3

需要替换以下substitute layer：

1. tiny test VLM -> 真实Stage 2 checkpoint
2. mock image/messages -> 真实CURV/MIMIC-derived GRPO JSONL与图像
3. smoke-only reward -> 全部original external CheXbert/RadGraph reward plugins
4. 2个short generations -> 8个最长1536 tokens的generations
5. CPU 1 step -> original distributed full-training schedule、vLLM与ZeRO-3

删除smoke-only reward前，必须安装并验证medical reward dependency与checkpoint。
否则group内reward variance可能再次变为零，从而失去policy-gradient signal。

Transformers 5.14.1会警告此unit-test checkpoint中缺少`lm_head.weight`，并在
load时对其进行初始化。只有因为该model被明确视为wiring substitute，这种行为才
可以接受；该事实也记录在result JSON中。

</details>
