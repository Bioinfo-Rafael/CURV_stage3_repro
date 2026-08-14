# CURV Stage 3 最小 GRPO 复现

[日本語](README.ja.md) | [English](README.en.md) | **中文** | [语言选择](README.md)

## 快速运行

```bash
cd /Users/cls-lab/Git/CURV_stage3_repro && bash scripts/run_stage3_smoke.sh
```

首次运行时，请先按照下方的[环境配置与运行](#环境配置与运行)创建`.venv`。

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

