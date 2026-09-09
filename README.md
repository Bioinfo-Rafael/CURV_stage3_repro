# CURV Stage 3 minimal GRPO reproduction

環境構築・実画像・全ローカルテストの統合RUN_GUIDE:
[日本語](../CXR_LLM/docs/RUN_GUIDE.ja.md) | [English](../CXR_LLM/docs/RUN_GUIDE.en.md) | [简体中文](../CXR_LLM/docs/RUN_GUIDE.zh-CN.md)

[日本語版](README.ja.md) | [English version](README.en.md) | [中文版](README.zh-CN.md)

## Quick run / クイック実行

Location: `/Users/cls-lab/Git/LinGu/CURV_stage3_repro`.
Stage 1/2・CLARITYの実画像は隣接CXR_LLM、NIH/MIMIC実画像はCXR_GRNで扱います。
ここは既存のsynthetic GRPO smokeのままです（MIMIC実画像の学習成功とは扱いません）。
Real-image pipelines live in the sibling CXR_LLM/CXR_GRN; this Stage 3 smoke
continues to use synthetic input. The relocated `.venv` can be reused.

```bash
cd /Users/cls-lab/Git/LinGu/CURV_stage3_repro && bash scripts/run_stage3_smoke.sh
```

初回のみ、各言語版READMEのsetup手順で`.venv`を作成してください。
For the first run, create `.venv` using the setup instructions in your chosen README.

公式CURVに未追跡root `.DS_Store`だけがある場合／if only an untracked root
`.DS_Store` blocks the upstream clean check, run from the same directory:
`.venv/bin/python scripts/run_stage3_local_smoke.py`.
Tracked upstream files and smoke assertions remain unchanged.

CURV Stage 3 の GRPO control flow を、mock data と tiny Qwen2.5-VL を使って
1 optimizer step 実行する最小再現です。公式 `CURV/` checkout の tracked file
は変更しません。

This is a minimal, one-optimizer-step reproduction of the CURV Stage 3 GRPO
control flow using mock data and a tiny Qwen2.5-VL. It does not modify any
tracked file in the official `CURV/` checkout.

詳細なセットアップ、実行方法、再現範囲、実測結果については、上記の言語版を
参照してください。

Choose a language above for full setup instructions, scope, fidelity notes, and
measured verification details.
