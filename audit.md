# CURV Stage 3 code audit

Audit target: official CURV commit `f8bf7d0ad5f3c26e9336f118c78b6264887c947b`.
The sibling `CURV/` checkout is treated as an immutable upstream snapshot. Line
numbers below refer to that commit.

## A. Stage 3で本来行われる処理

The highest-priority source is
`CURV/training/scripts/run_grpo_training.sh:42-120`, especially the preserved
`swift rlhf --rlhf_type grpo` command at lines 75-120. Its intended flow is:

1. Load the Stage 2 Qwen2.5-VL checkpoint and the messages-style CXR JSONL.
2. Apply the CXR system prompt from `training/prompts/prompt_cxr.txt`.
3. Generate 8 sampled completions per prompt (`temperature=1.0`, `top_p=0.9`,
   `top_k=50`, `max_completion_length=1536`), with vLLM in the full recipe.
4. Score each completion with format, CheXbert F1, RadGraph F1, and two
   uncertain-label RadGraph variants listed in the preserved ms-swift command.
5. Form group-relative normalized advantages and optimize a GRPO policy
   objective. The recipe requests full-model training, LR `1e-7`, batch 4 per
   device, accumulation 2, two training processes, and DeepSpeed ZeRO-3.

The shell wrapper's active Python command (lines 42-72) differs from the
preserved command: it lists only `format_cxr,f1_chexbert_cxr,radgraph_f1_cxr`.
For recipe intent, this audit follows the preserved ms-swift command as required.

## B. 現在のrepositoryに実装されている処理

- `prompts/prompt_cxr.txt` and `prompts/cxr_prompts.py:45-77` define the requested
  `<findings>`, `<thinking>`, `<impression>` structure.
- `reward_functions/format_reward.py:15-88` implements a real binary format
  check: exact ordered tags, exactly once, with no outside content.
- `accuracy_reward.py` implements a lightweight term-matching reward rather than
  the CheXbert reward named by the original recipe.
- `coherence_reward.py` is intended to combine medical entity/dependency tools.
- `data_utils.py:14-418` defines a local JSON/JSONL dataset path, conversation
  construction, text tokenization, and optional image processing.
- `trainers/grpo_trainer.py:145-178` sketches generate -> reward -> KL -> policy
  loss -> total loss around Hugging Face `Trainer`.

## C. placeholder / mock / incomplete implementation

- `grpo_trainer.py:123-143,432-449`: vLLM is a `MockVLLMEngine`; it returns the
  same hard-coded tagged response instead of model inference.
- `grpo_trainer.py:299-324`: reference KL is a zero tensor placeholder.
- `grpo_trainer.py:326-344`: policy loss is `-rewards.mean()` and is not
  connected to policy log-probabilities or model parameters. Backward cannot
  constitute GRPO training.
- `grpo_trainer.py:293-297`: ground truth extraction always returns `None`.
- `grpo_trainer.py:204-237`: the non-vLLM generator creates only one response per
  input row and does not implement the configured group of 8.
- `data_utils.py:148-156`: absent image processing falls back to an all-zero
  tensor; the repository trainer does not establish a Qwen2.5-VL processor path.

These components are not used as evidence of a successful reproduction.

## D. repository内部で矛盾しているinterface

- `run_grpo_training.sh:42-72` passes `--dataset_path`, `--reward_functions`,
  `--use_vllm true`, `--train_type`, `--torch_dtype`, and many other names not
  accepted by `train_grpo.py:27-96`. The Python CLI instead defines `--dataset`,
  `--reward_funcs`, `--vllm_enable` (a flag), and `--sft_type`.
- `train_grpo.py:22` imports nonexistent `training.utils.logging_utils`.
- `train_grpo.py:24,242` imports/calls nonexistent `prepare_dataset`; the actual
  utility exports `prepare_training_data` at `data_utils.py:394`.
- `reward_functions/__init__.py:12-13` imports nonexistent
  `f1_chexbert_reward.py` and `radgraph_f1_reward.py`, so importing the package
  fails before `FormatRewardCXR` can be obtained.
- The registry in `reward_functions/__init__.py:15-20` exposes three names while
  the active shell command and preserved ms-swift command request different
  reward sets.
- `train_grpo.py:205-208` constructs `RewardConfig(debug_mode=..., log_path=...)`,
  but `grpo_config.py:185-193` declares neither field. Lines 147-160 also read
  several nonexistent `RewardConfig` attributes.
- `GRPOConfig` has no `num_generations`, `temperature`, `top_p`, `top_k`, or
  `max_completion_length` fields used by the original recipe.
- `grpo_trainer.py:114-115` calls `_setup_vllm()` before `self.logger` is assigned
  at line 121; both its try and except paths access that missing attribute.
- `model_utils.py` uses causal-LM/tokenizer auto classes, not an explicit
  Qwen2.5-VL multimodal model + processor path.
- `grpo_config.py:95` points at `prompts/system_prompt_cxr.txt`, which does not
  exist; the tracked file is `training/prompts/prompt_cxr.txt`.

No upstream file was patched to hide or absorb these inconsistencies.

## E. 今回どの実装を再利用するか

Direct official reuse:

- The system prompt is read verbatim at runtime from
  `CURV/training/prompts/prompt_cxr.txt`.
- `FormatRewardCXR` is executed directly from the tracked `base_reward.py` and
  `format_reward.py`. An isolated import namespace avoids executing the broken
  package `__init__.py`; no reward logic is copied or changed.
- Hyperparameter intent and the fidelity comparison come from the preserved
  ms-swift command in `run_grpo_training.sh`.

Equivalent external implementation:

- TRL 1.9.2 `GRPOTrainer` supplies multiple generation, Qwen2.5-VL multimodal
  processing, policy/reference token log-probabilities, group-relative
  normalization, clipped ratio objective, KL penalty, backward, and optimizer.
  `AuditedGRPOTrainer` only records intermediate tensors and gradients.

Mock/substituted:

- `trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration` substitutes for
  the unavailable Stage 2 checkpoint. It is a random/unit-test model, not Stage 2.
- One synthetic image and one mock reference report substitute for CXR data.
- `smoke_only_index_reward` is explicitly **SMOKE-ONLY REWARD; NOT PART OF
  CURV**. It guarantees within-group variance because random 16-token outputs do
  not satisfy the official format reward.
- One CPU optimizer step and 2 generations substitute for the distributed run.

## Library-path attempts

1. ms-swift/vLLM was unavailable in the original environment and the preserved
   recipe requires CUDA, two inference GPUs, and six 60-GiB training GPUs. It was
   not installed on this Apple Silicon host.
2. TRL 0.19.1 was tested first but its GRPO tokenizer path was text-only and did
   not forward image tensors, so it could not satisfy the VLM-input requirement.
3. Current TRL 1.9.2 was then tested and explicitly supports multimodal fields.
4. An MPS run reached image generation and the differentiable loss but PyTorch
   2.7.1 failed during backward with a scalar-type assertion. The verified run
   uses CPU; no algorithm or data-flow stage was removed.

## Upstream integrity

The smoke script checks `git status --short` and `git diff` both before and after
training, and fails if the snapshot is dirty. It also disables bytecode writes
while directly importing reward source. The verified result records
`upstream_modified: false`.

