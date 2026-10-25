# FinMMEval 2026, Task 1: Financial Exam Question Answering

University team project on the CLEF 2026 FinMMEval Lab, Task 1. We reproduce the organizers' LoRA baselines for two small language models, evaluate them on held-out and test data, and look for ways to improve them.

[![CLEF 2026](https://img.shields.io/badge/CLEF-2026-1f6feb)](https://github.com/mbzuai-nlp/CLEF-2026-FinMMEval-Lab)
[![Task](https://img.shields.io/badge/Task%201-Financial%20Exam%20QA-2da44e)](https://github.com/mbzuai-nlp/CLEF-2026-FinMMEval-Lab)
[![Models](https://img.shields.io/badge/Models-Qwen2.5--3B%20%7C%20Ministral--3--3B-8957e5)](#-models)
[![Method](https://img.shields.io/badge/Method-LoRA-orange)](#-method)
[![License](https://img.shields.io/badge/Use-Academic-lightgrey)](#-disclaimer)

Authors: Lukáš Švihura, Ján Hrbál, Dominik Keil 

Organizers' repository: https://github.com/mbzuai-nlp/CLEF-2026-FinMMEval-Lab

## Task

The system receives one multiple-choice question from a finance or accounting exam, together with its options, and returns the label of the correct option. Explanations and confidence are not evaluated. The official task languages are English, Chinese, Arabic and Hindi.

Most financial NLP work is done in English, and Arabic in particular is still little explored. Terminology, scripts and exam conventions differ between languages, so a model that is strong in English may not transfer. The questions also require finance vocabulary, calculation and conceptual understanding, not only general language skill.

## Goals

1. Understand the task and the data, collect relevant sources, and create appropriate training and testing data.
2. Replicate and test the official LoRA baselines on Qwen2.5-3B-Instruct and Ministral-3-3B-Instruct.
3. Improve data quality and consistency, and explore additional languages where appropriate.
4. Investigate approaches to improve baseline performance and, where beneficial, develop alternative models.
5. Document methodology, setup, implementation and results for reproducibility.

## Documentation

| Document | Content |
|---|---|
| `docs/report.pdf` | Full project report: methodology, experiments, results and discussion (to be added) |
| `README.md` (this file) | Overview and usage |
| `README_official_training.md` | Organizers' README for the training code, copied for reference |
| `README_test_data.md` | Description of the LLM-labeled public test set |

The report describes the whole process in detail. This README is a short overview.

## Data

### Training and validation data (official)

`artifacts/public_task1_train.jsonl` is produced by the organizers' `prepare_public_task1_data.py` from the public FinMMEval collection. It has 9,624 examples in six languages, with a `split` field (`train` or `validation`).

| Language | Total | Train | Validation | Sources |
|---|---:|---:|---:|---|
| English | 4,456 | 4,221 | 235 | BhashaBench-EN, CFA/CPA-EN |
| Hindi | 4,000 | 3,795 | 205 | BhashaBench-HI |
| Arabic | 523 | 501 | 22 | Arabic accounting, Arabic business |
| Greek | 268 | 262 | 6 | Plutus |
| Spanish | 230 | 220 | 10 | FLARE-ES |
| Chinese | 147 | 142 | 5 | CFA/CPA-ZH |
| Total | 9,624 | 9,141 | 483 | |

Greek and Spanish are not official task languages but are part of the public collection. We keep them as extra training data. The organizers' leaderboard dev sets and hidden test sets are never used for training. Statistics are in `artifacts/public_task1_stats.json`.

### Test data (unofficial labels)

`test_data_public/task1_final_test_public_data_label_by_LLMs.jsonl` contains the 800 public test questions (200 each in English, Chinese, Arabic and Hindi). The organizers released them without answers, so the labels were produced by three LLMs (Claude, GPT, Gemini), compared, and the disagreements re-solved. Details are in `README_test_data.md`.

These labels are model-generated, not the official answer key. Some source questions are ambiguous or broken, and label accuracy has not been measured. Treat test scores as approximate. The Chinese questions are mostly translations of the English ones, so the two languages are not independent.

## Evaluation setup

- Training uses only rows with `split == "train"`. Validation rows are used for the validation loss only.
- No checkpoint is chosen on validation: the final-step adapter is used.
- The LLM-labeled test set is used for evaluation only and never for training.
- Scoring uses the logits of the option letters at the first answer position, so there is no text parsing and no invalid output.
- Accuracy is reported overall, per language and per source.

## Setup

Run all commands from the project root (`task1_training_baselines/`).

```
pip install torch transformers peft datasets accelerate pandas
```

Ministral-3 needs a recent `transformers` version with Mistral3 support. A GPU is recommended. A CPU run is possible but slow (`--device cpu`, optionally `--dtype bf16` to halve RAM use).

Expected layout:

```
task1_training_baselines/
  artifacts/
    public_task1_train.jsonl
    public_task1_stats.json
    qwen25_3b_task1_lora/final_adapter/
    ministral3_3b_2512_task1_lora/final_adapter/
  test_data_public/
    task1_final_test_public_data_label_by_LLMs.jsonl
  results/
  evaluate_models.py
  train_qwen_lora.py
  prepare_public_task1_data.py
```

## Training

Training follows the organizers' configuration (LoRA r=16, alpha=32, dropout 0.05, bf16, 2 epochs, effective batch size 16, learning rate 2e-4, max length 1024).

```
python train_qwen_lora.py \
  --train-file artifacts/public_task1_train.jsonl \
  --output-dir artifacts/qwen25_3b_task1_lora \
  --model-name Qwen/Qwen2.5-3B-Instruct \
  --bf16 --per-device-train-batch-size 2 --gradient-accumulation-steps 8 --num-train-epochs 2

python train_qwen_lora.py \
  --train-file artifacts/public_task1_train.jsonl \
  --output-dir artifacts/ministral3_3b_2512_task1_lora \
  --model-name mistralai/Ministral-3-3B-Instruct-2512 \
  --bf16 --per-device-train-batch-size 2 --gradient-accumulation-steps 8 --num-train-epochs 2
```

## Evaluation: usage of `evaluate_models.py`

The script evaluates the base models and/or their LoRA adapters. Models are loaded one after another, never at once.

```
# 1) Check paths, data and tokenizers. No model weights are loaded, so it is safe on a weak PC.
python evaluate_models.py --mode validation --dry-run

# 2) Smoke test: 2 examples per language (results go to results/validation_smoke/)
python evaluate_models.py --mode validation --model qwen --smoke
python evaluate_models.py --mode validation --model ministral --smoke

# 3) Full run
python evaluate_models.py --mode validation --model both --variant lora
python evaluate_models.py --mode test --model both --variant both
```

| Option | Meaning |
|---|---|
| `--mode` | `validation` (split `validation` of the official file) or `test` (LLM-labeled test set, all rows) |
| `--model` | `qwen`, `ministral` or `both` |
| `--variant` | `base` (no adapter), `lora`, or `both` |
| `--smoke [N]` | N examples per language (default 2) |
| `--dry-run` | Check paths, data and tokenizers only |
| `--device`, `--dtype` | `auto`, `cuda`, `cpu`; `auto`, `bf16`, `fp16`, `fp32` |
| `--data`, `--split`, `--qwen-adapter`, `--ministral-adapter`, `--out-dir` | Override paths and filters |

Output goes to `results/<mode>/`: `eval_<model>_<variant>.json` (summary and per-question predictions) and `comparison.json` (all runs side by side).

## Google Colab

Training and the baseline evaluations were run on Google Colab because of hardware limits. The notebooks are in `notebooks/` and document the actual runs. The same code can be run locally with the commands above.

## Results

Accuracy on the held-out validation split and on the LLM-labeled test set (to be filled in from `results/`).

| Model | Validation | Test (LLM labels) |
|---|---:|---:|
| Qwen2.5-3B, base | TBD | TBD |
| Qwen2.5-3B, LoRA | TBD | TBD |
| Ministral-3-3B, base | TBD | TBD |
| Ministral-3-3B, LoRA | TBD | TBD |

Per-language and per-source tables are in the report. The English CFA/CPA part of the validation split has only 22 questions, so differences there are within noise.

## Attribution

The training code and the baseline setup are copied from the organizers' repository (`task1_training`): https://github.com/mbzuai-nlp/CLEF-2026-FinMMEval-Lab/tree/main/task1_training. The original authors retain all rights to that code and documentation; local changes are listed in `NOTICE.md`. Questions belong to their original owners. Citations:

- Xie et al., Overview of FinMMEval 2026: Multilingual and Multimodal Financial Evaluation, Springer LNCS, 2026. https://doi.org/10.1007/978-3-032-39150-6_27
- Xie et al., Overview of FinMMEval 2026 Task 1: Multilingual Financial Multiple-Choice Question Answering, CLEF 2026 Working Notes, CEUR-WS Vol. 4283. https://ceur-ws.org/Vol-4283/paper184.pdf
- Devane et al., BhashaBench V1, 2025. https://arxiv.org/abs/2510.25409
