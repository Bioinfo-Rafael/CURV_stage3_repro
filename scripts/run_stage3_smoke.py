#!/usr/bin/env python3
"""Run one real CURV Stage 3-shaped GRPO optimizer step on mock data.

The policy/model and data are substitutes. The format reward is loaded directly
from the untouched CURV checkout. Optimization is delegated to TRL's production
GRPOTrainer; the subclass below only records evidence and does not replace its
objective, generation, backward, or optimizer implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import types
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from PIL import Image, ImageDraw
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    set_seed,
)
from trl import GRPOConfig, GRPOTrainer


REPRO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CURV_ROOT = REPRO_ROOT.parent / "CURV"
RESULT_PATH = REPRO_ROOT / "outputs" / "stage3_smoke_result.json"
MOCK_DATA_PATH = REPRO_ROOT / "mock_data" / "mock_cxr.jsonl"
MOCK_IMAGE_PATH = REPRO_ROOT / "mock_data" / "synthetic_cxr.png"
CONFIG_PATH = REPRO_ROOT / "configs" / "smoke_config.json"
EXPECTED_UPSTREAM_COMMIT = "f8bf7d0ad5f3c26e9336f118c78b6264887c947b"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_synthetic_cxr(path: Path) -> None:
    """Create a deterministic, obviously synthetic 224x224 grayscale CXR mock."""
    if path.exists():
        return
    image = Image.new("L", (224, 224), color=18)
    draw = ImageDraw.Draw(image)
    draw.ellipse((34, 30, 108, 194), fill=88, outline=145, width=2)
    draw.ellipse((116, 30, 190, 194), fill=82, outline=145, width=2)
    draw.ellipse((87, 78, 140, 178), fill=112)
    draw.line((112, 18, 112, 204), fill=165, width=3)
    draw.ellipse((48, 150, 100, 180), fill=130)
    image.save(path)


def load_official_format_reward(curv_root: Path):
    """Load CURV's tracked reward files directly without executing broken __init__.py.

    The upstream package __init__ imports two files absent from the repository.
    An isolated package namespace lets Python resolve `.base_reward` while the
    executed class body remains exactly the tracked upstream implementation.
    """
    reward_dir = curv_root / "training" / "reward_functions"
    # Import the tracked source without leaving __pycache__ in the upstream snapshot.
    sys.dont_write_bytecode = True
    package_name = "_curv_official_reward_functions"
    package = types.ModuleType(package_name)
    package.__path__ = [str(reward_dir)]
    sys.modules[package_name] = package

    def load_module(module_name: str, path: Path):
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load official CURV module: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    load_module(f"{package_name}.base_reward", reward_dir / "base_reward.py")
    format_module = load_module(f"{package_name}.format_reward", reward_dir / "format_reward.py")
    return format_module.FormatRewardCXR(weight=1.0)


def completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        item = completion[-1]
        if isinstance(item, dict):
            content = item.get("content", "")
            if isinstance(content, str):
                return content
    return str(completion)


class RewardRecorder:
    def __init__(self, official_format_reward):
        self.official = official_format_reward
        self.completions: list[str] = []
        self.format_values: list[float] = []
        self.smoke_values: list[float] = []

    def official_format_cxr(self, completions, **_: Any) -> list[float]:
        """Thin adapter around the unmodified official FormatRewardCXR."""
        texts = [completion_text(item) for item in completions]
        values = self.official(texts)
        self.completions = texts
        self.format_values = [float(value) for value in values]
        return values

    def smoke_only_index_reward(self, completions, **_: Any) -> list[float]:
        """SMOKE-ONLY REWARD -- NOT PART OF CURV.

        Alternating scores guarantee non-zero within-group variance even when a
        random unit-test model emits equally malformed reports. This reward tests
        GRPO wiring only and has no medical or semantic meaning.
        """
        values = [float(index % 2) for index in range(len(completions))]
        self.smoke_values = values
        return values


class AuditedGRPOTrainer(GRPOTrainer):
    """TRL GRPOTrainer with read-only instrumentation for smoke assertions."""

    evidence: dict[str, Any]

    def __init__(self, *args, **kwargs):
        self.evidence = {}
        super().__init__(*args, **kwargs)

    def _generate_and_score_completions(self, inputs):
        output = super()._generate_and_score_completions(inputs)
        self.evidence["advantages"] = output["advantages"].detach().float().cpu().tolist()
        self.evidence["completion_token_ids"] = output["completion_ids"].detach().cpu().tolist()
        self.evidence["completion_mask"] = output["completion_mask"].detach().cpu().tolist()
        pixel_values = output.get("pixel_values")
        self.evidence["pixel_values_present"] = pixel_values is not None
        self.evidence["pixel_values_finite"] = bool(
            pixel_values is not None and torch.isfinite(pixel_values).all().item()
        )
        self.evidence["image_grid_thw_present"] = output.get("image_grid_thw") is not None
        return output

    def _policy_logps_for_audit(self, model, inputs):
        prompt_ids = inputs["prompt_ids"]
        completion_ids = inputs["completion_ids"]
        prompt_mask = inputs["prompt_mask"]
        completion_mask = inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logps, _, _ = self._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            completion_ids.size(1),
            compute_entropy=False,
            compute_aux_loss=False,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            num_images=inputs.get("num_images"),
            pixel_attention_mask=inputs.get("pixel_attention_mask"),
            spatial_shapes=inputs.get("spatial_shapes"),
            num_tiles=inputs.get("num_tiles"),
            image_sizes=inputs.get("image_sizes"),
            token_type_ids=inputs.get("token_type_ids"),
            mm_token_type_ids=inputs.get("mm_token_type_ids"),
            image_position_ids=inputs.get("image_position_ids"),
        )
        mask = completion_mask.bool()
        selected = logps[mask].detach().float().cpu()
        self.evidence["policy_logprobs"] = selected.tolist()
        self.evidence["policy_logprobs_finite"] = bool(
            selected.numel() > 0 and torch.isfinite(selected).all().item()
        )
        advantages = inputs["advantages"]
        if advantages.dim() == 1:
            advantages = advantages.unsqueeze(1)
        old_logps = inputs.get("old_per_token_logps")
        old_logps = logps.detach() if old_logps is None else old_logps
        ratio = torch.exp(logps - old_logps)
        clipped_ratio = torch.clamp(ratio, 1 - self.epsilon_low, 1 + self.epsilon_high)
        policy_per_token = -torch.min(ratio * advantages, clipped_ratio * advantages)
        policy_loss = ((policy_per_token * completion_mask).sum(-1) / completion_mask.sum(-1)).mean()
        self.evidence["policy_loss"] = float(policy_loss.detach().float().cpu().item())
        trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
        policy_grads = torch.autograd.grad(
            policy_loss, [parameter for _, parameter in trainable], allow_unused=True
        )
        nonzero_policy_grad_norms = {
            name: float(gradient.detach().float().norm().cpu().item())
            for (name, _), gradient in zip(trainable, policy_grads)
            if gradient is not None
            and math.isfinite(gradient.detach().float().norm().cpu().item())
            and gradient.detach().float().norm().cpu().item() > 0
        }
        self.evidence["policy_only_nonzero_gradient_norms"] = nonzero_policy_grad_norms
        self.evidence["policy_only_nonzero_gradient"] = bool(nonzero_policy_grad_norms)
        if inputs.get("ref_per_token_logps") is not None:
            ref = inputs["ref_per_token_logps"]
            per_token_kl = torch.exp(ref - logps) - (ref - logps) - 1
            selected_kl = per_token_kl[mask].detach().float().cpu()
            self.evidence["kl_values"] = selected_kl.tolist()
            self.evidence["kl"] = float(selected_kl.mean().item())
            self.evidence["kl_finite"] = bool(torch.isfinite(selected_kl).all().item())
            kl_loss = ((per_token_kl * completion_mask).sum(-1) / completion_mask.sum(-1)).mean()
            self.evidence["weighted_kl_loss"] = float((self.beta * kl_loss).detach().float().cpu().item())

    def _compute_loss(self, model, inputs):
        self._policy_logps_for_audit(model, inputs)
        loss = super()._compute_loss(model, inputs)
        self.evidence["loss"] = float(loss.detach().float().cpu().item())
        self.evidence["loss_finite"] = bool(torch.isfinite(loss).item())
        return loss

    def training_step(self, model, inputs, num_items_in_batch):
        output = super().training_step(model, inputs, num_items_in_batch)
        grad_norms = {}
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                norm = parameter.grad.detach().float().norm().cpu().item()
                if math.isfinite(norm) and norm > 0:
                    grad_norms[name] = float(norm)
        self.evidence["nonzero_grad_norms"] = grad_norms
        self.evidence["backward_ok"] = True
        self.evidence["nonzero_gradient"] = bool(grad_norms)
        return output


def read_mock_sample(curv_root: Path) -> tuple[dict[str, Any], Image.Image]:
    rows = [json.loads(line) for line in MOCK_DATA_PATH.read_text().splitlines() if line.strip()]
    if len(rows) != 1:
        raise AssertionError(f"Smoke run requires exactly one mock row, got {len(rows)}")
    sample = rows[0]
    official_prompt = (curv_root / "training" / "prompts" / "prompt_cxr.txt").read_text().strip()
    messages = sample["messages"]
    if messages[0]["content"] != "__LOAD_VERBATIM_FROM_CURV_training/prompts/prompt_cxr.txt__":
        raise AssertionError("Mock system prompt source marker is missing")
    messages[0]["content"] = official_prompt
    image = Image.open(MOCK_DATA_PATH.parent / sample["image"]).convert("RGB")
    return sample, image


def git_state(curv_root: Path) -> tuple[str, str, bool]:
    import subprocess

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=curv_root, text=True).strip()
    status = subprocess.check_output(["git", "status", "--short"], cwd=curv_root, text=True)
    diff = subprocess.check_output(["git", "diff"], cwd=curv_root, text=True)
    return commit, status, not status and not diff


def assert_finite(values: list[float], label: str) -> None:
    if not values or not all(math.isfinite(value) for value in values):
        raise AssertionError(f"{label} must be non-empty and finite: {values}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--curv-root", type=Path, default=DEFAULT_CURV_ROOT)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args()
    curv_root = args.curv_root.resolve()
    config = json.loads(args.config.read_text())

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    make_synthetic_cxr(MOCK_IMAGE_PATH)
    set_seed(config["seed"])
    random.seed(config["seed"])

    upstream_commit_before, upstream_status_before, upstream_clean_before = git_state(curv_root)
    if not upstream_clean_before:
        raise AssertionError(f"CURV checkout is not clean before run:\n{upstream_status_before}")

    sample, image = read_mock_sample(curv_root)
    prompt_messages = sample["messages"][:-1]
    dataset = Dataset.from_list([{"prompt": prompt_messages, "image": image}])
    print("DATA LOAD OK")

    processor = AutoProcessor.from_pretrained(
        config["model_id"],
        trust_remote_code=True,
    )
    print("PROCESSOR OK")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config["model_id"],
        dtype=torch.float32,
        trust_remote_code=True,
    )
    print("MODEL LOAD OK")

    official_format = load_official_format_reward(curv_root)
    recorder = RewardRecorder(official_format)

    use_mps = config["device_preference"] == "mps_then_cpu" and torch.backends.mps.is_available()
    training_args = GRPOConfig(
        output_dir=str(REPRO_ROOT / "outputs" / "trl_work"),
        max_steps=1,
        per_device_train_batch_size=config["num_generations"],
        gradient_accumulation_steps=1,
        generation_batch_size=config["num_generations"],
        num_generations=config["num_generations"],
        max_completion_length=config["max_completion_length"],
        temperature=config["temperature"],
        top_p=config["top_p"],
        top_k=config["top_k"],
        learning_rate=config["learning_rate"],
        beta=config["beta"],
        epsilon=config["epsilon"],
        loss_type="grpo",
        scale_rewards="group",
        reward_weights=[config["format_reward_weight"], config["smoke_only_reward_weight"]],
        use_vllm=False,
        use_cpu=not use_mps,
        seed=config["seed"],
        data_seed=config["seed"],
        disable_dropout=True,
        max_grad_norm=1.0,
        logging_steps=1,
        log_completions=False,
        save_strategy="no",
        eval_strategy="no",
        report_to="none",
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
    )

    trainer = AuditedGRPOTrainer(
        model=model,
        reward_funcs=[recorder.official_format_cxr, recorder.smoke_only_index_reward],
        args=training_args,
        train_dataset=dataset,
        processing_class=processor,
    )

    tracked_name = "lm_head.weight"
    named_parameters = dict(trainer.model.named_parameters())
    if tracked_name not in named_parameters:
        tracked_name = next(name for name, param in named_parameters.items() if param.requires_grad)
    tracked_parameter = named_parameters[tracked_name]
    before = tracked_parameter.detach().clone().cpu()

    trainer.train()
    after = tracked_parameter.detach().clone().cpu()
    max_parameter_delta = float((after - before).abs().max().item())
    parameter_changed = not torch.equal(before, after)

    evidence = trainer.evidence
    reward_values = [
        config["format_reward_weight"] * format_value
        + config["smoke_only_reward_weight"] * smoke_value
        for format_value, smoke_value in zip(recorder.format_values, recorder.smoke_values)
    ]
    reward_tensor = torch.tensor(reward_values, dtype=torch.float32)
    reward_std = float(reward_tensor.std(unbiased=True).item())
    advantages = [float(value) for value in evidence.get("advantages", [])]

    upstream_commit_after, upstream_status_after, upstream_clean_after = git_state(curv_root)
    result = {
        "pipeline": "CURV Stage 3 GRPO control-flow smoke reproduction",
        "implementation": "TRL GRPOTrainer 1.9.2 (external formal implementation)",
        "model": config["model_id"],
        "model_architecture": model.config.model_type,
        "model_num_parameters_loaded": sum(parameter.numel() for parameter in model.parameters()),
        "model_is_substitute": True,
        "model_warning": "MODEL MOCK / SMALL SUBSTITUTE; not a CURV Stage 2 checkpoint",
        "model_load_note": "Transformers 5.14.1 reports lm_head.weight missing in this unit-test checkpoint and initializes it; this further limits model fidelity",
        "device": str(trainer.accelerator.device),
        "dataset": "mock synthetic CXR",
        "num_prompts": 1,
        "num_generations": config["num_generations"],
        "max_completion_length": config["max_completion_length"],
        "generation_ok": len(recorder.completions) == config["num_generations"],
        "completion_texts": recorder.completions,
        "completion_token_ids": evidence.get("completion_token_ids", []),
        "vlm_pixel_values_present": evidence.get("pixel_values_present", False),
        "vlm_pixel_values_finite": evidence.get("pixel_values_finite", False),
        "vlm_image_grid_thw_present": evidence.get("image_grid_thw_present", False),
        "official_rewards_used": ["format_cxr (FormatRewardCXR loaded from official tracked file)"],
        "official_format_reward_values": recorder.format_values,
        "smoke_only_rewards_used": ["smoke_only_index_reward (SMOKE-ONLY REWARD; NOT PART OF CURV)"],
        "smoke_only_reward_values": recorder.smoke_values,
        "reward_values": reward_values,
        "reward_mean": float(reward_tensor.mean().item()),
        "reward_std": reward_std,
        "reward_vector_finite": bool(torch.isfinite(reward_tensor).all().item()),
        "advantages": advantages,
        "advantages_finite": bool(advantages and all(math.isfinite(value) for value in advantages)),
        "policy_logprobs": evidence.get("policy_logprobs", []),
        "policy_logprobs_finite": evidence.get("policy_logprobs_finite", False),
        "kl": evidence.get("kl"),
        "kl_values": evidence.get("kl_values", []),
        "kl_finite": evidence.get("kl_finite", False),
        "beta": config["beta"],
        "clip_epsilon": config["epsilon"],
        "policy_loss": evidence.get("policy_loss"),
        "policy_only_nonzero_gradient": evidence.get("policy_only_nonzero_gradient", False),
        "policy_only_nonzero_gradient_parameter_count": len(
            evidence.get("policy_only_nonzero_gradient_norms", {})
        ),
        "policy_only_nonzero_gradient_norms": evidence.get("policy_only_nonzero_gradient_norms", {}),
        "weighted_kl_loss": evidence.get("weighted_kl_loss"),
        "total_loss": evidence.get("loss"),
        "loss_finite": evidence.get("loss_finite", False),
        "backward_ok": evidence.get("backward_ok", False),
        "nonzero_gradient": evidence.get("nonzero_gradient", False),
        "nonzero_gradient_parameter_count": len(evidence.get("nonzero_grad_norms", {})),
        "nonzero_gradient_norms": evidence.get("nonzero_grad_norms", {}),
        "optimizer_step": trainer.state.global_step == 1,
        "optimizer_global_step": trainer.state.global_step,
        "tracked_parameter": tracked_name,
        "max_parameter_delta": max_parameter_delta,
        "parameter_changed": parameter_changed,
        "official_prompt_sha256": sha256(curv_root / "training" / "prompts" / "prompt_cxr.txt"),
        "official_format_reward_sha256": sha256(
            curv_root / "training" / "reward_functions" / "format_reward.py"
        ),
        "upstream_commit_expected": EXPECTED_UPSTREAM_COMMIT,
        "upstream_commit_before": upstream_commit_before,
        "upstream_commit_after": upstream_commit_after,
        "upstream_status_before": upstream_status_before,
        "upstream_status_after": upstream_status_after,
        "upstream_modified": not upstream_clean_after,
    }

    checks = {
        "GENERATION OK": result["generation_ok"],
        "NUM GENERATIONS >= 2": result["num_generations"] >= 2,
        "VLM PIXEL INPUT OK": result["vlm_pixel_values_present"] and result["vlm_pixel_values_finite"],
        "CURV FORMAT REWARD OK": len(result["official_format_reward_values"]) == result["num_generations"],
        "REWARD VECTOR FINITE": result["reward_vector_finite"],
        "REWARD STD > 0": result["reward_std"] > 0,
        "ADVANTAGES FINITE": result["advantages_finite"],
        "POLICY LOGPROBS FINITE": result["policy_logprobs_finite"],
        "KL FINITE": result["kl_finite"],
        "GRPO LOSS FINITE": result["loss_finite"],
        "POLICY-ONLY NONZERO GRADIENT": result["policy_only_nonzero_gradient"],
        "BACKWARD OK": result["backward_ok"],
        "NONZERO GRADIENT EXISTS": result["nonzero_gradient"],
        "OPTIMIZER STEP OK": result["optimizer_step"],
        "PARAMETER CHANGED": result["parameter_changed"],
        "UPSTREAM UNMODIFIED": not result["upstream_modified"],
    }
    result["checks"] = checks
    RESULT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")

    for label, passed in checks.items():
        print(f"{label}: {'OK' if passed else 'FAILED'}")
    print(f"RESULT: {RESULT_PATH}")
    failed = [label for label, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(f"Smoke checks failed: {failed}")


if __name__ == "__main__":
    main()
