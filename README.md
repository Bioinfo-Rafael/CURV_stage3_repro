# CURV Stage 3 — GRPOを1 step動かす最小再現

**synthetic画像とtiny Qwen2.5-VLで、生成→reward→GRPO loss→backward→optimizer更新を実行する**repositoryです。公式CURVのpromptとformat rewardを再利用し、学習経路が本当に通るかをhard assertionで確認します。

実画像のStage 1/2は[CXR_LLM](../CXR_LLM/README.md)、画像分類は[CXR_GRN](../CXR_GRN/README.md)の担当です。ここでの成功はMIMIC実画像学習や医学的性能の検証を意味しません。

[日本語](README.ja.md) | [English](README.en.md) | [简体中文](README.zh-CN.md)

[できること](#capabilities) · [構成](#layout) · [実行方法](#run) · [実装上の違い](#implementation) · [既存詳細・履歴](#archive)

<a id="capabilities"></a>

## 1. 何を確認するか

| 処理 | Smokeでの確認 |
|---|---|
| 入力・生成 | synthetic画像1 promptから2 responses生成、VLM pixel入力 |
| Reward | 公式format reward＋明示したsmoke-only index reward、有限値・group std > 0 |
| GRPO | advantage、policy/reference logprob、KL、clipped objectiveが有限 |
| 学習 | 非ゼロgradient、backward、optimizer step、parameter changed |
| 公式code | 指定したcleanなCURV checkoutが実行前後で未変更 |

CPUで1 step実行します。tinyモデル全体の更新であり、CXR_LLMのnorm 1 parameterだけのsmokeとは学習範囲が異なります。

<a id="layout"></a>

## 2. Directory構成

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

## 3. 実行方法

このrepositoryのrootから実行します。既存環境はそのまま使用できます。

### 初回の環境準備

Python 3.11、torch 2.7.1、Transformers 5.14.1、TRL 1.9.2で検証しています。`.venv`がない場合のみ作成してください。

```bash
conda create -p .venv python=3.11 pip -y
.venv/bin/python -m pip install -r requirements/requirements.txt
```

tinyモデル／processorを初めて取得する場合はnetworkが必要です。下記のoffline変数を外して初回実行し、cache後はそのまま使えます。MIMICのdownloadは不要です。

### Synthetic GRPO smoke

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 bash scripts/run_stage3_smoke.sh \
  --curv-root ../CXR_LLM/models/CURV
```

`--curv-root`は公式prompt/rewardを読むcheckoutです。隣接`CURV/README.md`を編集した現在の構成では、同commitの未変更コピー`CXR_LLM/models/CURV`を指定します。データやモデルを複製する操作ではありません。

この参照先がない別環境では、cleanな公式commit `f8bf7d0ad5f3c26e9336f118c78b6264887c947b` を用意しpathを置き換えます。指定先のtracked変更をreset／退避して実行する必要はありません。別のcleanなcheckoutを指定してください。未追跡root `.DS_Store`だけがある場合に限り、既存`run_stage3_local_smoke.py --curv-root <path>`がFinder metadataを一時保管・復元できます。

### 成功判定と出力

終了code 0に加え、`GENERATION OK`、`REWARD STD > 0`、`GRPO LOSS FINITE`、`BACKWARD OK`、`OPTIMIZER STEP OK`、`PARAMETER CHANGED`、`UPSTREAM UNMODIFIED`等がすべて通ることを確認します。

結果は[outputs/stage3_smoke_result.json](outputs/stage3_smoke_result.json)。synthetic画像がなければdeterministicに生成します。policy lossが0でもgradientが0とは限らないため、policy-onlyとtotal-lossのgradientを別々に検査します。

<a id="implementation"></a>

## 4. 公式実装との違い・未対応範囲

| 要素 | 公式recipe／本格実験 | このsmoke |
|---|---|---|
| Model/input | Stage 2 checkpoint＋実データ | tiny Qwen2.5-VL＋synthetic 1 prompt |
| Optimizer実装 | ms-swift系GRPO recipe | TRL 1.9.2＋観測用instrumentation |
| Reward | format＋CheXbert／RadGraph等 | 公式format＋smoke-only variance reward |
| 生成数・長さ | 8 generations、最大1,536 tokens | 2 generations、最大16 tokens |
| 更新 | 分散full schedule、learning rate `1e-7` | CPU 1 step、`1e-4` |
| 公式source | 参照元 | 変更せずimport。clean／未変更assertionを維持 |

`lm_head.weight`がないtiny checkpointの初期化warningも記録します。modelは`MODEL MOCK / SMALL SUBSTITUTE`、補助rewardは`SMOKE-ONLY REWARD; NOT PART OF CURV`として扱います。医学的rewardの依存／checkpoint、実画像messages、正式Stage 2モデルを揃える前に、full reproductionと呼んではいけません。

[監査記録](audit.md)と末尾の詳細対応表に、libraryの試行・代替範囲・実験設定を残しています。全repositoryを再実行する入口は[統合RUN_GUIDE](../CXR_LLM/docs/RUN_GUIDE.ja.md#stage3)です。

<a id="archive"></a>

## 付録：既存本文・全assertion・設定比較・移動履歴

以下は整理前のREADME全文をそのまま保存しています。絶対pathや旧default起動例は履歴です。現在のcheckout指定は上の実行方法を使ってください。

<details>
<summary>整理前のREADME全文 / Previous README (verbatim) / 原README全文</summary>

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

</details>
