# CURV Stage 3 — A One-Step GRPO Reproduction

This repository **runs generation → reward → GRPO loss → backward → optimizer update using a synthetic image and tiny Qwen2.5-VL**. It reuses the official CURV prompt and format reward, with hard assertions proving that the training path executes.

Real-image Stage 1/2 belongs to [CXR_LLM](../CXR_LLM/README.md); image classification belongs to [CXR_GRN](../CXR_GRN/README.md). Success here is not real-MIMIC training or medical validation.

[日本語](README.ja.md) | [English](README.en.md) | [简体中文](README.zh-CN.md)

[Capabilities](#capabilities) · [Layout](#layout) · [Run](#run) · [Implementation differences](#implementation) · [Original details/history](#archive)

<a id="capabilities"></a>

## 1. What It Checks

| Step | Smoke verification |
|---|---|
| Input/generation | Two responses from one synthetic-image prompt, with VLM pixel input |
| Reward | Official format + explicit smoke-only index reward; finite values, group std > 0 |
| GRPO | Finite advantages, policy/reference log probabilities, KL, clipped objective |
| Training | Nonzero gradients, backward, optimizer step, measured parameter change |
| Official code | The selected clean CURV checkout stays unchanged before/after execution |

It runs one CPU step, updating the full tiny model. This differs from CXR_LLM's single-norm-parameter smoke.

<a id="layout"></a>

## 2. Directory Layout

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

## 3. How to Run

Run from this repository root. Reuse an existing environment.

### First-Time Setup

Verified versions: Python 3.11, torch 2.7.1, Transformers 5.14.1, TRL 1.9.2. Create `.venv` only if absent.

```bash
conda create -p .venv python=3.11 pip -y
.venv/bin/python -m pip install -r requirements/requirements.txt
```

Initial tiny-model/processor fetching requires network access. Omit the offline variables below on first use, then reuse the cached model. No MIMIC download is needed.

### Synthetic GRPO Smoke

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 bash scripts/run_stage3_smoke.sh \
  --curv-root ../CXR_LLM/models/CURV
```

`--curv-root` selects the checkout providing the official prompt/reward. Since the sibling `CURV/README.md` is now locally edited, use the unchanged same-commit copy at `CXR_LLM/models/CURV`. This command does not copy data or models.

On another machine without this checkout, supply a clean official commit `f8bf7d0ad5f3c26e9336f118c78b6264887c947b` and replace the path. Do not reset or temporarily hide tracked edits to pass the check; select another clean checkout. Only when an untracked root `.DS_Store` is the sole obstacle may the existing `run_stage3_local_smoke.py --curv-root <path>` preserve and restore Finder metadata.

### Success and Output

Require exit code 0 and every assertion, including `GENERATION OK`, `REWARD STD > 0`, `GRPO LOSS FINITE`, `BACKWARD OK`, `OPTIMIZER STEP OK`, `PARAMETER CHANGED`, and `UPSTREAM UNMODIFIED`.

Evidence is written to [outputs/stage3_smoke_result.json](outputs/stage3_smoke_result.json). The synthetic image is created deterministically if absent. Zero policy loss need not mean zero gradients; policy-only and total-loss gradients are checked separately.

<a id="implementation"></a>

## 4. Implementation Differences and Limits

| Component | Official recipe / full experiment | This smoke |
|---|---|---|
| Model/input | Stage 2 checkpoint + real data | Tiny Qwen2.5-VL + one synthetic prompt |
| Optimization implementation | ms-swift GRPO recipe | TRL 1.9.2 + observation instrumentation |
| Reward | Format + CheXbert/RadGraph variants | Official format + smoke-only variance reward |
| Generations/length | 8 generations, up to 1,536 tokens | 2 generations, up to 16 tokens |
| Update | Distributed full schedule; learning rate `1e-7` | One CPU step; `1e-4` |
| Official source | Reference implementation | Imported unchanged, with strict clean/unchanged assertions |

The missing `lm_head.weight` initialization warning is recorded. The model is labeled `MODEL MOCK / SMALL SUBSTITUTE`, and the auxiliary reward is `SMOKE-ONLY REWARD; NOT PART OF CURV`. Medical-reward dependencies/checkpoints, real-image messages, and the actual Stage 2 model are required before claiming full reproduction.

The [audit](audit.md) and preserved tables below retain library attempts, substitutions, and experimental settings. For all repositories, use the existing [integrated RUN_GUIDE](../CXR_LLM/docs/RUN_GUIDE.en.md#stage3).

<a id="archive"></a>

## Appendix: Original Text, Assertions, Settings, and Relocation History

The complete previous README is preserved verbatim below. Absolute paths and default-launch examples are historical; use the explicit checkout selection above for the current setup.

<details>
<summary>整理前のREADME全文 / Previous README (verbatim) / 原README全文</summary>

# CURV Stage 3 minimal GRPO reproduction

[日本語](README.ja.md) | **English** | [中文](README.zh-CN.md) | [Language index](README.md)

## Quick run

All four repositories now live under `/Users/cls-lab/Git/LinGu/`.
Real-image Stage 1/2 and CLARITY smoke runs belong to the sibling CXR_LLM;
real NIH/MIMIC MRGL runs belong to CXR_GRN. This Stage 3 remains a synthetic-image
GRPO smoke: manual MIMIC images and Stage 1/2 teacher JSON are not automatically
used here. The relocated `.venv` is usable; no environment recreation or model
redownload is needed. See the existing [integrated RUN_GUIDE](../CXR_LLM/docs/RUN_GUIDE.en.md).

```bash
cd /Users/cls-lab/Git/LinGu/CURV_stage3_repro && bash scripts/run_stage3_smoke.sh
```

For the first run, create `.venv` using the [setup commands](#setup-and-run) below.

The command above requires a clean official CURV checkout. If only an untracked
root `.DS_Store` blocks the check, run `.venv/bin/python scripts/run_stage3_local_smoke.py`
from the same directory. It temporarily preserves/restores Finder metadata without
changing images, tracked upstream files, or existing assertions.

This project executes one real optimizer step through the **CURV Stage 3 GRPO
control flow** using mock data, while leaving the sibling official `CURV/`
checkout unchanged.

## What is reproduced

- official CURV CXR system prompt -> synthetic image + user prompt
- 2 sampled responses from a tiny Qwen2.5-VL architecture
- official CURV `FormatRewardCXR`
- group-relative reward mean/std and normalized advantages
- policy and frozen-reference completion-token log-probabilities
- TRL's clipped GRPO objective with finite KL
- backward, non-zero gradient, optimizer step, and a measured parameter change

## What is NOT reproduced

- paper performance or reported numbers
- the full dataset or medical validity
- the real Stage 2 checkpoint or its training
- CheXbert/RadGraph rewards and their external checkpoints
- full-scale 8-GPU/vLLM/DeepSpeed ZeRO-3 training
- long 1536-token reports or the full training schedule

The model is always reported as **MODEL MOCK / SMALL SUBSTITUTE**. The
index-based variance reward is always reported as **SMOKE-ONLY REWARD; NOT PART
OF CURV**.

## Component mapping

| Reproduction | Official CURV source | Role | Fidelity |
|---|---|---|---|
| `mock_cxr.jsonl` + synthetic PNG | original GRPO messages JSONL | training input | mock |
| runtime-loaded prompt | `training/prompts/prompt_cxr.txt` | system prompt | official, verbatim |
| runtime-loaded `FormatRewardCXR` | `training/reward_functions/format_reward.py` | format reward | official, direct execution |
| TRL 1.9.2 `GRPOTrainer` | preserved ms-swift GRPO recipe | optimization | equivalent formal implementation |
| tiny Qwen2.5-VL | Stage 2 checkpoint path | policy/reference VLM | small substitute |
| `smoke_only_index_reward` | none | guarantee reward variance | smoke-only substitute |
| audit instrumentation subclass | none | save intermediate evidence | observation only |

## Original vs smoke settings

| Setting | CURV original recipe | Smoke run |
|---|---:|---:|
| model | Stage 2 Qwen2.5-VL checkpoint | tiny Qwen2.5-VL test model (6,288,704 parameters loaded here) |
| prompts | full dataset | 1 mock prompt |
| generations/prompt | 8 | 2 |
| max completion length | 1536 | 16 |
| temperature / top-p / top-k | 1.0 / 0.9 / 50 | same |
| train type | full | full tiny-model update |
| learning rate | `1e-7` | `1e-4` so one float32 step is measurable |
| rewards | format + CheXbert + RadGraph variants | official format + labeled smoke-only variance reward |
| accelerator | 8 GPUs, vLLM, ZeRO-3 intent | CPU (MPS backward was incompatible) |
| optimizer steps | full epoch | 1 |

## Setup and run

The verified environment is Python 3.11 on Apple Silicon:

```bash
conda create -p .venv python=3.11 pip -y
.venv/bin/python -m pip install -r requirements/requirements.txt
```

Run the complete one-step smoke test with one command from this directory:

```bash
bash scripts/run_stage3_smoke.sh
```

The script creates the deterministic 224x224 synthetic image if absent and
writes measured evidence to `outputs/stage3_smoke_result.json`. A Hugging Face
network request is needed only until the tiny model/processor is cached.

## Hard assertions

The command exits non-zero unless all of the following are true:

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

See `audit.md` for the upstream-code audit, placeholders, contradictory
interfaces, library-path attempts, and reuse decisions.

With two symmetric normalized advantages, the first-step clipped policy loss
value can be exactly `0.0`; that does not imply a zero derivative because the two
different sampled token sequences have different log-probability derivatives.
The script therefore asserts both the policy-only gradient and the gradient from
the actual total-loss backward independently.

## Moving to real CURV Stage 3

Replace only the substituted layers:

1. tiny test VLM -> actual Stage 2 checkpoint
2. mock image/messages -> actual CURV/MIMIC-derived GRPO JSONL and images
3. smoke-only reward -> all original external CheXbert/RadGraph reward plugins
4. 2 short generations -> 8 generations of up to 1536 tokens
5. one CPU step -> original distributed full-training schedule, vLLM, and ZeRO-3

The medical reward dependencies and checkpoints must be installed and validated
before removing the smoke-only reward; otherwise zero group variance can again
remove the policy-gradient signal.

Transformers 5.14.1 warns that `lm_head.weight` is absent from this unit-test
checkpoint and initializes it at load time. This is acceptable only because the
model is explicitly a wiring substitute, and is recorded in the result JSON.

</details>
