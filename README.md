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

The system receives one multiple-choice question from a finance or accounting exam, together with its options, and returns the label of the correct option.  Explanations and confidence are not evaluated. The official task languages are English, Chinese, Arabic and Hindi. Model should be capable of answering questions in multiple languages.

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

## Setup and training
 
Training and data preparation are the organizers' code and are documented in the README inside the training folder (see `README_official_training.md`). We used it unchanged (LoRA, bf16, 2 epochs). The organizers' commands use their own machine paths (`/home/zhuohanx/...` and a `task1_training/` prefix), so replace those with your own Python and project paths.
 
Requirements: Python with `torch`, `transformers`, `peft`, `datasets`, `accelerate` and `pandas`. Ministral-3 needs a recent `transformers` with Mistral3 support. A GPU is recommended; a CPU run is possible but slow (`--device cpu`, optionally `--dtype bf16`).


## Attribution

The training code and the baseline setup are copied from the organizers' repository (`task1_training`): https://github.com/mbzuai-nlp/CLEF-2026-FinMMEval-Lab/tree/main/task1_training. The original authors retain all rights to that code and documentation; local changes are listed in `NOTICE.md`. Questions belong to their original owners. Citations:

- Xie et al., Overview of FinMMEval 2026: Multilingual and Multimodal Financial Evaluation, Springer LNCS, 2026. https://doi.org/10.1007/978-3-032-39150-6_27
- Xie et al., Overview of FinMMEval 2026 Task 1: Multilingual Financial Multiple-Choice Question Answering, CLEF 2026 Working Notes, CEUR-WS Vol. 4283. https://ceur-ws.org/Vol-4283/paper184.pdf
- Devane et al., BhashaBench V1, 2025. https://arxiv.org/abs/2510.25409
