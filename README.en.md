# CURV Stage 3 minimal GRPO reproduction

[日本語](README.ja.md) | **English** | [中文](README.zh-CN.md) | [Language index](README.md)

## Quick run

```bash
cd /Users/cls-lab/Git/CURV_stage3_repro && bash scripts/run_stage3_smoke.sh
```

For the first run, create `.venv` using the [setup commands](#setup-and-run) below.

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
