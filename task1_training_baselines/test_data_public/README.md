# FinMMEval 2026 Task 1: Labeled Public Test Set (Unofficial)

Answer labels for the public test set of **CLEF 2026 FinMMEval, Task 1**: 800 multilingual financial multiple-choice questions. The labels were produced by three large language models (Claude, GPT and Gemini) and combined into one final file.

> [!IMPORTANT]
> These are **model-generated labels, not the official answer key.** They are for building and evaluating our own baselines and are not part of the official competition.

## Contents

- [Why we did this](#why-we-did-this)
- [Repository structure](#repository-structure)
- [The data](#the-data)
- [Format](#format)
- [How the labels were made](#how-the-labels-were-made)
- [Possible improvements](#possible-improvements)
- [Recommended use](#recommended-use)
- [Acknowledgements](#acknowledgements)

## Why we did this

The organizers released the Task 1 public test questions [without answers](https://github.com/mbzuai-nlp/CLEF-2026-FinMMEval-Lab/tree/main/task1_dev_leaderboard/dev_sets). To build a stronger baseline and measure it on these questions, we needed labels, so we labeled them ourselves using three models.

## Repository structure

```text
.
├── README.md
├── task1_final_test_public_labeled_final.jsonl     # final labels, 800 rows
└── cache_LLM_labels/                               # source labels, kept for reference
    ├── claude_task1_final_test_labeled.jsonl
    ├── gpt_task1_final_test_labeled.jsonl
    └── gemini_task1_final_test_labeled.jsonl
```

| File | Description |
|---|---|
| `task1_final_test_public_labeled_final.jsonl` | **Final labels.** All 800 questions: English, Chinese, Arabic, Hindi. |
| `cache_LLM_labels/claude_task1_final_test_labeled.jsonl` | Source labels from Claude |
| `cache_LLM_labels/gpt_task1_final_test_labeled.jsonl` | Source labels from GPT |
| `cache_LLM_labels/gemini_task1_final_test_labeled.jsonl` | Source labels from Gemini |

The files in `cache_LLM_labels/` show how the final labels were built. They differ slightly in formatting and are not meant to be loaded directly. Use the final file.

## The data

The four public test files contain 200 questions each:

| Source file | Language | Topics |
|---|---|---|
| `english_task1_final_test_public.jsonl` | English | CFA, CMA, FRM, actuarial, US tax (EA exam), ESG |
| `chinese_task1_final_test_public.jsonl` | Chinese | Mostly translations of the English questions, in a different order |
| `arabic_task1_final_test_public.jsonl` | Arabic | Introductory, government (Saudi) and cost accounting |
| `hindi_task1_final_test_public.jsonl` | Hindi | Economics: micro, macro, money and banking, Indian economy |

Questions have 2 to 5 options. Arabic rows 51–70 are true/false.

## Format

Each line of the final file is one JSON object:

```json
{
  "id": "en-task1-final-hidden-001",
  "source": "english_task1_final_test_public",
  "language": "english",
  "prompt": "Read the following multiple-choice question and return only the correct option letter.\n\nQuestion:\n...\n\nOptions:\nA. ...\nB. ...\n\nAnswer:",
  "answer": "A",
  "num_choices": 4,
  "split": "test"
}
```

| Field | Type | Meaning |
|---|---|---|
| `id` | string | Question id from the source file. Hindi ids were renamed from `hindi-test-NNN` to `hi-task1-final-test-NNN` to match the other languages. |
| `source` | string | Source file name |
| `language` | string | `english`, `chinese`, `arabic` or `hindi` |
| `prompt` | string | The question and options, inserted unchanged into a fixed instruction template |
| `answer` | string | The correct option letter (A–E) |
| `num_choices` | int | Number of options (2–5) |
| `split` | string | Always `test` |

Rows are ordered English 1–200, Chinese 1–200, Arabic 1–200, Hindi 1–200.

## How the labels were made

1. **Independent labeling.** Claude, GPT and Gemini each labeled all 800 questions.
2. **Validation.** Each model's output was checked against the original files: every question present, question and option text unchanged, and every answer a valid option letter.
3. **Combining.** The three label sets were combined into one final answer per question. Where the models differed, the question was re-solved. Each Chinese question is a translation of an English one, so every such pair was given the same answer.
4. **Final checks.** The final file contains all 800 questions, with no rows removed. Every field except `answer` matches the original data exactly, all ids are unique, and every answer is a valid option.

## Possible improvements

Some questions are ambiguous or broken in the source data: tables or case text are missing, options are cut off or duplicated, or no option matches the correct calculation. The final file keeps them all. Reviewing or removing these questions could improve label quality.

## Recommended use

> [!WARNING]
> **Don't train and evaluate on the same questions.** This is the competition's public test set. If you train on these labels and then evaluate on these or overlapping questions, scores will be inflated. Use the file for either training or evaluation, or split it into separate parts first.

If label accuracy matters for your results, hand-check a random sample of 50–100 rows to get a measured accuracy figure.

## Acknowledgements

Questions come from the [CLEF 2026 FinMMEval Lab](https://github.com/mbzuai-nlp/CLEF-2026-FinMMEval-Lab). All rights to the original questions belong to their owners. Only the labels in this repository are our own work.
