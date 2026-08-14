# CURV Stage 3 minimal GRPO reproduction

[日本語版](README.ja.md) | [English version](README.en.md) | [中文版](README.zh-CN.md)

## Quick run / クイック実行

```bash
cd /Users/cls-lab/Git/CURV_stage3_repro && bash scripts/run_stage3_smoke.sh
```

初回のみ、各言語版READMEのsetup手順で`.venv`を作成してください。
For the first run, create `.venv` using the setup instructions in your chosen README.

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
