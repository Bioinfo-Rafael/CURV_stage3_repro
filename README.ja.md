# CURV Stage 3 最小 GRPO 再現

Stage 1/2・CLARITY・NIH/MIMIC・unit testsを含む手順:
[統合RUN_GUIDE](../CXR_LLM/docs/RUN_GUIDE.ja.md)。
未追跡Finder `.DS_Store`だけがupstream clean-checkを妨げる場合は
`.venv/bin/python scripts/run_stage3_local_smoke.py` で一時退避・実行・復元できる。
既存assertionやtracked変更は隠さない。

**日本語** | [English](README.en.md) | [中文](README.zh-CN.md) | [言語選択](README.md)

## クイック実行

4 repositoriesは`/Users/cls-lab/Git/LinGu/`配下に移動しました。
Stage 1/2・CLARITYの実画像smokeは隣接するCXR_LLM、NIH/MIMIC実画像MRGLは
CXR_GRNが担当します。このStage 3は引き続きsynthetic画像のGRPO smokeであり、
手動MIMIC画像やStage 1/2の教師JSONを自動的に入力するものではありません。
移動済みの`.venv`は使用可能で、再作成やモデル再downloadは不要です。

```bash
cd /Users/cls-lab/Git/LinGu/CURV_stage3_repro && bash scripts/run_stage3_smoke.sh
```

初回のみ、下記の[セットアップ手順](#セットアップと実行)で`.venv`を作成してください。

上記は公式CURVがcleanな場合のcommandです。未追跡root `.DS_Store`のみがある
本機では、同じdirectoryから`.venv/bin/python scripts/run_stage3_local_smoke.py`
を使用します。画像の実体・公式CURVのtracked file・既存assertionは変更しません。

このプロジェクトは、隣接する公式 `CURV/` checkout を変更せず、mock data を
使って **CURV Stage 3 GRPO control flowを実際に1 optimizer step実行**します。

## 再現しているもの

- 公式CURV CXR system prompt -> synthetic image + user prompt
- tiny Qwen2.5-VL architectureからの2件のsampled response
- 公式CURV `FormatRewardCXR`
- group内rewardのmean/stdと正規化advantage
- policyおよび固定referenceのcompletion-token log probability
- TRLのclipped GRPO objectiveと有限なKL
- backward、非ゼロgradient、optimizer step、parameter変化の実測

## 再現していないもの

- 論文性能や報告された数値
- full datasetおよびmedical validity
- 実際のStage 2 checkpointやその学習
- CheXbert/RadGraph rewardと外部checkpoint
- 8-GPU/vLLM/DeepSpeed ZeRO-3によるfull-scale training
- 1536-token reportやfull training schedule

使用モデルは常に **MODEL MOCK / SMALL SUBSTITUTE** として記録します。
reward varianceを保証するindex-based rewardは、常に
**SMOKE-ONLY REWARD; NOT PART OF CURV** として記録します。

## 公式CURVとの対応

| 再現側 | 公式CURV側 | 役割 | 再現度 |
|---|---|---|---|
| `mock_cxr.jsonl` + synthetic PNG | original GRPO messages JSONL | training input | mock |
| runtimeで読み込むprompt | `training/prompts/prompt_cxr.txt` | system prompt | 公式、verbatim |
| runtimeで読み込む`FormatRewardCXR` | `training/reward_functions/format_reward.py` | format reward | 公式、直接実行 |
| TRL 1.9.2 `GRPOTrainer` | 保存済みms-swift GRPO recipe | optimization | 正式な同等外部実装 |
| tiny Qwen2.5-VL | Stage 2 checkpoint path | policy/reference VLM | small substitute |
| `smoke_only_index_reward` | なし | reward varianceの保証 | smoke-only substitute |
| audit instrumentation subclass | なし | 中間値の保存 | 観測のみ |

## Original設定とsmoke設定

| 設定 | CURV original recipe | Smoke run |
|---|---:|---:|
| model | Stage 2 Qwen2.5-VL checkpoint | tiny Qwen2.5-VL test model（この環境では6,288,704 parameters） |
| prompts | full dataset | mock prompt 1件 |
| generations/prompt | 8 | 2 |
| max completion length | 1536 | 16 |
| temperature / top-p / top-k | 1.0 / 0.9 / 50 | 同一 |
| train type | full | tiny modelのfull update |
| learning rate | `1e-7` | float32の1 stepで変化を測れるよう`1e-4` |
| rewards | format + CheXbert + RadGraph variants | 公式format + 明示したsmoke-only variance reward |
| accelerator | 8 GPUs、vLLM、ZeRO-3の意図 | CPU（MPS backwardは非互換） |
| optimizer steps | full epoch | 1 |

## セットアップと実行

検証済み環境はApple Silicon上のPython 3.11です。

```bash
conda create -p .venv python=3.11 pip -y
.venv/bin/python -m pip install -r requirements/requirements.txt
```

このdirectoryから、次の1 commandで完全な1-step smoke testを実行します。

```bash
bash scripts/run_stage3_smoke.sh
```

synthetic imageが存在しない場合、scriptがdeterministicな224×224画像を生成します。
実測結果は `outputs/stage3_smoke_result.json` に保存されます。tiny modelと
processorがcacheされるまではHugging Faceへのnetwork accessが必要です。

## Hard assertions

次のうち1つでも満たさない場合、commandはnon-zeroで終了します。

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

upstream codeの監査、placeholder、interface不整合、library pathの試行、再利用方針は
`audit.md` を参照してください。

正規化された2つのadvantageが対称な場合、最初のstepのclipped policy loss値は
ちょうど`0.0`になり得ます。ただし2つの異なるsampled token sequenceでは
log-probabilityのderivativeが異なるため、これはzero gradientを意味しません。
そのためscriptは、policy-only gradientと、実際のtotal-loss backwardによる
gradientを独立にassertします。

## 実際のCURV Stage 3へ移行する場合

次のsubstitute layerを差し替えます。

1. tiny test VLM -> 実際のStage 2 checkpoint
2. mock image/messages -> 実際のCURV/MIMIC-derived GRPO JSONLと画像
3. smoke-only reward -> original external CheXbert/RadGraph reward plugin一式
4. 2件のshort generation -> 最大1536 tokensの8 generations
5. CPU 1 step -> original distributed full-training schedule、vLLM、ZeRO-3

smoke-only rewardを削除する前に、medical rewardのdependencyとcheckpointを導入し、
実際に動作することを検証してください。そうしないとgroup内reward varianceが再び
ゼロになり、policy-gradient signalが消える可能性があります。

Transformers 5.14.1は、このunit-test checkpointに`lm_head.weight`が存在しない
というwarningを出し、load時に初期化します。この挙動を許容できるのは、modelを
明示的にwiring substituteとして扱っているためです。この事実もresult JSONに
記録しています。
