> **Attribution / Source notice**
>
> This README and the accompanying training code were copied from the official
> CLEF-2026 FinMMEval Lab repository:
> https://github.com/mbzuai-nlp/CLEF-2026-FinMMEval-Lab/tree/main/task1_training
>
> It is included here for a school project. The original authors retain all
> rights to the code and documentation. Any local modifications are noted at
> the end of this file or in the accompanying `NOTICE.md`.

# Task 1 LoRA Baselines

This directory contains a lightweight training pipeline for public-data-only Task 1 baselines using:

- `Qwen/Qwen2.5-3B-Instruct`
- `mistralai/Ministral-3-3B-Instruct-2512`
- QLoRA / LoRA
- only public datasets from the FinMMEval collection

Policy note:

- The FinMMEval Hugging Face collection is the official public data release for Task 1.
- The original split names on dataset cards do not restrict participant usage; the released collection may be reorganized or re-split for training.
- The separate Task 1 leaderboard dev sets are organizer-held evaluation data and should not be mixed back into training.

Recommended environment on this machine:

- Python: `/home/zhuohanx/.conda/envs/finance/bin/python`

## 1. Prepare public training data

```bash
/home/zhuohanx/.conda/envs/finance/bin/python task1_training/prepare_public_task1_data.py
```

This writes:

- `task1_training/artifacts/public_task1_train.jsonl`
- `task1_training/artifacts/public_task1_stats.json`

## 2. Train a Qwen2.5-3B LoRA baseline

```bash
/home/zhuohanx/.conda/envs/finance/bin/python task1_training/train_qwen_lora.py \
  --train-file task1_training/artifacts/public_task1_train.jsonl \
  --output-dir task1_training/artifacts/qwen25_3b_task1_lora \
  --model-name Qwen/Qwen2.5-3B-Instruct \
  --bf16 \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --num-train-epochs 2
```

Official organizer baseline configuration used for the current public Task 1 leaderboards:

- Base model: `Qwen/Qwen2.5-3B-Instruct`
- Fine-tuning: LoRA
- Precision: `bf16`
- Epochs: `2`
- Training source: only the official public FinMMEval Hugging Face collection
- Excluded data: all organizer-held Task 1 leaderboard dev sets and all hidden test sets

## 3. Train a Ministral 3 3B LoRA baseline

`mistralai/Ministral-3-3B-Instruct-2512` requires a recent Transformers version with Mistral3 support. On this machine, the `finance` environment has been upgraded for this run; keep it separate from any vLLM-serving environment if dependency constraints matter.

```bash
/home/zhuohanx/.conda/envs/finance/bin/python task1_training/train_qwen_lora.py \
  --train-file task1_training/artifacts/public_task1_train.jsonl \
  --output-dir task1_training/artifacts/ministral3_3b_2512_task1_lora \
  --model-name mistralai/Ministral-3-3B-Instruct-2512 \
  --bf16 \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --num-train-epochs 2 \
  --logging-steps 10 \
  --eval-steps 100 \
  --save-steps 100
```

Current Hindi dev result from the public-data-only LoRA run:

- Model: `mistralai/Ministral-3-3B-Instruct-2512`
- Adapter: `task1_training/artifacts/ministral3_3b_2512_task1_lora/final_adapter`
- Training loss: `0.5216`
- Final validation loss: `0.6166`
- Hindi dev accuracy: `61/100 = 61%`

For comparison, the existing Qwen2.5-3B public baseline file scored `45/100 = 45%` on the same Hindi dev set.



## Attribution and Citations

This `task1_training/` directory was copied from the official CLEF-2026
FinMMEval Lab repository for use as a baseline in a school project:

- Source: https://github.com/mbzuai-nlp/CLEF-2026-FinMMEval-Lab/tree/main/task1_training

If you use this code or the FinMMEval Task 1 data, please cite the original
work. Only Task 1–relevant citations are listed below.

```bibtex
@inproceedings{FinMMEval2026,
  title = {Overview of {FinMMEval} 2026: Multilingual and Multimodal Financial Evaluation},
  author = {Xie, Zhuohan and Dai, Yuyang and Elbadry, Rania and Jani, Vanshikaa and Peng, Xueqing and Qian, Lingfei and Georgiev, Georgi and Dimitrov, Dimitar and Zhang, Fan and Huang, Jimin and Geng, Jiahui and Chen, Yankai and Yuan, Ye and Wu, Haolun and Wang, Yuxia and Koychev, Ivan and Stoyanov, Veselin and Song, Mingzi and Chen, Yu and Liu, Xue and Nakov, Preslav},
  booktitle = {Experimental IR Meets Multilinguality, Multimodality, and Interaction},
  series = {Lecture Notes in Computer Science},
  year = {2026},
  month = {September},
  pages = {427--441},
  doi = {10.1007/978-3-032-39150-6_27},
  url = {https://doi.org/10.1007/978-3-032-39150-6_27},
  publisher = {Springer},
}

@inproceedings{FinMMEvalTask1Overview2026,
  title = {Overview of {FinMMEval} 2026 Task 1: Multilingual Financial Multiple-Choice Question Answering},
  author = {Zhuohan Xie and Yuyang Dai and Rania Elbadry and Vanshikaa Jani and Georgi Georgiev and Dimitar Dimitrov and Fan Zhang and Xueqing Peng and Lingfei Qian and Jimin Huang and Jiahui Geng and Yankai Chen and Ye Yuan and Haolun Wu and Yuxia Wang and Ivan Koychev and Veselin Stoyanov and Mingzi Song and Yu Chen and Xue Liu and Preslav Nakov},
  booktitle = {CLEF 2026 Working Notes},
  series = {CEUR Workshop Proceedings},
  year = {2026},
  month = {September},
  pages = {2524--2532},
  volume = {4283},
  url = {https://ceur-ws.org/Vol-4283/paper184.pdf},
  publisher = {CEUR-WS.org},
}

@misc{devane2025bhashabenchv1,
  title = {{BhashaBench V1}: A Comprehensive Benchmark for the Quadrant of Indic Domains},
  author = {Vijay Devane and Mohd Nauman and Bhargav Patel and Aniket Mahendra Wakchoure and Yogeshkumar Sant and Shyam Pawar and Viraj Thakur and Ananya Godse and Sunil Patra and Neha Maurya and Suraj Racha and Nitish Kamal Singh and Ajay Nagpal and Piyush Sawarkar and Kundeshwar Vijayrao Pundalik and Rohit Saluja and Ganesh Ramakrishnan},
  year = {2025},
  eprint = {2510.25409},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
  url = {https://arxiv.org/abs/2510.25409},
}
