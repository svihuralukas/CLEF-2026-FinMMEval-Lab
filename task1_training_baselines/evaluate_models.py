#!/usr/bin/env python
"""
Unified evaluation of Qwen2.5-3B-Instruct and Ministral-3-3B-Instruct
(baseline and/or LoRA) for local execution.

Run from the project root (task1_training_baselines/):

  # Check paths, data and tokenizers WITHOUT loading model weights (lightweight)
  python evaluate_models.py --mode validation --dry-run

  # Smoke test: 2 examples per language, one model
  python evaluate_models.py --mode validation --model qwen --smoke

  # Full validation run, both LoRA models (loaded one after another, never at once)
  python evaluate_models.py --mode validation --model both --variant lora

  # Test set (LLM-labelled), baseline and LoRA for Ministral
  python evaluate_models.py --mode test --model ministral --variant both

Expected jsonl row format:
  {"id", "language", "source", "prompt", "answer", "num_choices", "split"}
If "prompt" is missing but "question" + "options" are present, the prompt is
built from LANGUAGE_PROMPTS below.
"""
import argparse
import collections
import gc
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

try:  # Windows consoles are often not UTF-8 (Hindi, Arabic, Chinese, ...)
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ============================================================
# CONFIG - all paths are relative to the project root
# ============================================================
ROOT = Path(__file__).resolve().parent

MODELS = {
    "qwen": {
        "base": "Qwen/Qwen2.5-3B-Instruct",
        # Directory containing adapter_config.json, or its parent holding final_adapter/
        "adapter": ROOT / "artifacts" / "qwen25_3b_task1_lora",
    },
    "ministral": {
        "base": "mistralai/Ministral-3-3B-Instruct-2512",
        "adapter": ROOT / "artifacts" / "ministral3_3b_2512_task1_lora",
    },
}

MODES = {
    "validation": {
        "data": ROOT / "artifacts" / "public_task1_train.jsonl",
        "split": "validation",
    },
    "test": {
        "data": ROOT / "test_data_public" / "task1_final_test_public_data_label_by_LLMs.jsonl",
        "split": None,  # None = use all rows
    },
}

RESULTS_DIR = ROOT / "results"

LANGUAGE_PROMPTS = {
    "hindi":   "निम्नलिखित प्रश्न पढ़ें और सही उत्तर चुनें। केवल विकल्प का अक्षर लिखें।",
    "chinese": "请阅读下面的选择题，并只返回正确选项对应的字母。",
    "arabic":  "اقرأ السؤال التالي واختر الإجابة الصحيحة فقط. أجب بحرف الخيار فقط.",
    "english": "Read the following multiple-choice question and return only the correct option letter.",
    "spanish": "Lee la siguiente pregunta de opción múltiple y devuelve solo la letra de la respuesta correcta.",
    "greek":   "Διάβασε την παρακάτω ερώτηση πολλαπλής επιλογής και επέστρεψε μόνο το γράμμα της σωστής απάντησης.",
}

LETTERS = "ABCDEF"


# ============================================================
# HELPERS
# ============================================================
def build_prompt_from_row(row):
    """Return row['prompt'] if present, otherwise build it from question + options + language."""
    if row.get("prompt"):
        return row["prompt"]
    lang = row.get("language", "english")
    header = LANGUAGE_PROMPTS.get(lang, LANGUAGE_PROMPTS["english"])
    option_lines = [f"{k}. {v}" for k, v in row["options"].items() if v]
    return (
        f"{header}\n\n"
        f"Question:\n{row['question']}\n\n"
        f"Options:\n" + "\n".join(option_lines) + "\n\nAnswer:"
    )


def load_rows(path, split=None):
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    if split is not None:
        rows = [r for r in rows if r.get("split") == split]
    for r in rows:
        gold = r.get("gold_answer") or r.get("answer") or r.get("label")
        if gold is None:
            raise ValueError(f"Row {r.get('id')} has no gold answer (gold_answer/answer/label).")
        r["_gold"] = str(gold).strip().upper()[:1]
    return rows


def smoke_subset(rows, per_language):
    """First N rows of each language - a quick test that still covers every language."""
    seen = collections.Counter()
    out = []
    for r in rows:
        lang = r.get("language", "unknown")
        if seen[lang] < per_language:
            out.append(r)
            seen[lang] += 1
    return out


def resolve_adapter(path):
    """Accept either the adapter directory itself or its parent containing final_adapter/."""
    path = Path(path)
    if (path / "adapter_config.json").exists():
        return path
    if (path / "final_adapter" / "adapter_config.json").exists():
        return path / "final_adapter"
    return None


def resolve_device(choice):
    if choice == "cpu":
        return "cpu"
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("--device cuda was requested, but CUDA is not available (check your torch install).")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def pick_dtype(device, choice):
    table = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    if choice != "auto":
        return table[choice]
    if device == "cpu":
        return torch.float32
    return torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16


def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# MODEL + EVALUATION
# ============================================================
def load_base_model(key, dtype, device):
    base = MODELS[key]["base"]
    kwargs = {"dtype": dtype}
    if device == "cuda":
        kwargs["device_map"] = "auto"
    if key == "ministral":
        # Imported here so Qwen still works with an older transformers version
        from transformers import Mistral3ForConditionalGeneration
        model = Mistral3ForConditionalGeneration.from_pretrained(base, **kwargs)
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(base, **kwargs)
    if device == "cpu":
        model.to("cpu")
    return model


@torch.no_grad()
def evaluate(model, tok, rows, letter_ids, label):
    print(f"\n{'=' * 70}\nEVAL: {label}  ({len(rows)} examples)\n{'=' * 70}")
    model.eval()
    device = model.device
    t0 = time.time()

    preds = []
    stat = collections.defaultdict(lambda: [0, 0])

    for i, r in enumerate(rows, 1):
        text = tok.apply_chat_template(
            [{"role": "user", "content": build_prompt_from_row(r)}],
            tokenize=False, add_generation_prompt=True,
        )
        x = tok(text, return_tensors="pt", add_special_tokens=False).to(device)
        logits = model(**x).logits[0, -1]

        n = int(r.get("num_choices") or len(r.get("options", {})) or 4)
        pred = LETTERS[int(torch.argmax(logits[letter_ids[:n]]))]
        gold = r["_gold"]
        ok = pred == gold

        for k in ("ALL", r.get("language", "unknown"), r.get("source", "unknown")):
            stat[k][0] += int(ok)
            stat[k][1] += 1

        preds.append({
            "id": r.get("id"), "language": r.get("language"), "source": r.get("source"),
            "pred": pred, "gold": gold, "ok": ok,
        })

        if i % 50 == 0 or i == len(rows):
            print(f"  {i}/{len(rows)}  ({time.time() - t0:.0f}s)")

    summary = {k: {"acc": round(c / n, 4), "correct": c, "n": n}
               for k, (c, n) in sorted(stat.items())}
    print(f"  ALL acc = {summary['ALL']['acc']:.4f}  ({summary['ALL']['correct']}/{summary['ALL']['n']})")
    return {"summary": summary, "preds": preds}


def run_model(key, variants, adapter_path, rows, dtype, device, tok, letter_ids):
    """Load the base model once; evaluate the baseline, then attach LoRA and evaluate that too."""
    print(f"\n>>> Loading {MODELS[key]['base']} (dtype={dtype}, device={device})")
    model = load_base_model(key, dtype, device)
    results = {}

    if "base" in variants:
        results["base"] = evaluate(model, tok, rows, letter_ids, f"{key} / baseline")

    if "lora" in variants:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter_path))
        print(f"LoRA adapter loaded from {adapter_path}")
        # Unify dtype - the base may be bf16 (FP8 dequantisation) while LoRA weights are fp16/fp32
        base_dtype = next(model.parameters()).dtype
        model = model.to(base_dtype)
        results["lora"] = evaluate(model, tok, rows, letter_ids, f"{key} / LoRA")

    del model
    free_memory()
    return results


# ============================================================
# MAIN
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate Qwen / Ministral (baseline + LoRA) locally.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--mode", choices=list(MODES), default="validation",
                   help="validation = artifacts/public_task1_train.jsonl (split=validation); "
                        "test = test_data_public/task1_final_test_public_data_label_by_LLMs.jsonl")
    p.add_argument("--model", choices=[*MODELS, "both"], default="both")
    p.add_argument("--variant", choices=["lora", "base", "both"], default="lora",
                   help="lora = with adapter, base = without adapter, both = evaluate both")
    p.add_argument("--smoke", nargs="?", const=2, type=int, default=None, metavar="N",
                   help="Smoke test: N examples per language (default 2). "
                        "Results go to results/<mode>_smoke/")
    p.add_argument("--dry-run", action="store_true",
                   help="Only check paths, data and tokenizers. Does not load model weights.")
    p.add_argument("--data", help="Override the data path (jsonl).")
    p.add_argument("--split", help="Override the split filter (e.g. 'test'); 'all' = no filter.")
    p.add_argument("--qwen-adapter", help="Override the path to the Qwen LoRA adapter.")
    p.add_argument("--ministral-adapter", help="Override the path to the Ministral LoRA adapter.")
    p.add_argument("--out-dir", help="Override the output directory.")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto",
                   help="On CPU, bf16 halves RAM usage (slower, slightly different numbers).")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = MODES[args.mode]

    data_path = Path(args.data) if args.data else cfg["data"]
    if args.split is None:
        split = cfg["split"]
    else:
        split = None if args.split.lower() == "all" else args.split

    keys = list(MODELS) if args.model == "both" else [args.model]
    variants = ["base", "lora"] if args.variant == "both" else [args.variant]

    adapter_roots = {
        "qwen": Path(args.qwen_adapter) if args.qwen_adapter else MODELS["qwen"]["adapter"],
        "ministral": Path(args.ministral_adapter) if args.ministral_adapter else MODELS["ministral"]["adapter"],
    }

    out_name = args.mode + ("_smoke" if args.smoke is not None else "")
    out_dir = Path(args.out_dir) if args.out_dir else RESULTS_DIR / out_name

    device = resolve_device(args.device)
    dtype = pick_dtype(device, args.dtype)

    print(f"mode={args.mode}  model={keys}  variant={variants}  smoke={args.smoke}  dry_run={args.dry_run}")
    print(f"data={data_path}  split={split}")
    print("GPU:", torch.cuda.get_device_name(0) if device == "cuda" else "CPU", "| dtype:", dtype)

    # ---------- path checks (before anything heavy is loaded) ----------
    problems = []
    adapters = {}
    if not data_path.exists():
        problems.append(f"Data file not found: {data_path}")
    if "lora" in variants:
        for k in keys:
            resolved = resolve_adapter(adapter_roots[k])
            if resolved is None:
                problems.append(
                    f"LoRA adapter for '{k}' not found in: {adapter_roots[k]}\n"
                    f"      Expected adapter_config.json + adapter_model.safetensors there "
                    f"(or inside a final_adapter/ subfolder)."
                )
            else:
                adapters[k] = resolved
                print(f"[{k}] adapter: {resolved}")
    if problems:
        print("\nPROBLEMS:")
        for pr in problems:
            print("  -", pr)
        raise SystemExit(1)

    # ---------- data ----------
    rows = load_rows(data_path, split=split)
    if not rows:
        raise SystemExit("No rows left after filtering. Check --split / --mode.")
    if args.smoke is not None:
        rows = smoke_subset(rows, args.smoke)
    langs = collections.Counter(r.get("language", "unknown") for r in rows)
    print(f"Examples: {len(rows)}  | languages: {dict(langs)}")

    # ---------- tokenizers + letter token ids ----------
    toks, letter_ids = {}, {}
    for k in keys:
        toks[k] = AutoTokenizer.from_pretrained(MODELS[k]["base"])
        if toks[k].pad_token is None:
            toks[k].pad_token = toks[k].eos_token
        letter_ids[k] = [toks[k].encode(l, add_special_tokens=False)[0] for l in LETTERS]
        print(f"[{k}] token ids for {LETTERS}: {letter_ids[k]}")

    # ---------- dry run stops here ----------
    if args.dry_run:
        r0 = rows[0]
        k0 = keys[0]
        text = toks[k0].apply_chat_template(
            [{"role": "user", "content": build_prompt_from_row(r0)}],
            tokenize=False, add_generation_prompt=True,
        )
        n_tok = len(toks[k0](text, add_special_tokens=False)["input_ids"])
        print(f"\nSample (id={r0.get('id')}, gold={r0['_gold']}, tokens={n_tok}):\n{text[:600]}")
        print("\nDRY RUN OK - paths, data and tokenizers are fine. No model weights were loaded.")
        return

    # ---------- evaluation (models are loaded one at a time) ----------
    out_dir.mkdir(parents=True, exist_ok=True)
    all_runs = {}  # "qwen/lora" -> {"summary": ..., "preds": ...}

    for k in keys:
        res = run_model(k, variants, adapters.get(k), rows, dtype, device, toks[k], letter_ids[k])
        for variant, r in res.items():
            all_runs[f"{k}/{variant}"] = r
            path = out_dir / f"eval_{k}_{variant}.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(r, f, indent=2, ensure_ascii=False)
            print(f"Saved: {path}")

    # ---------- comparison ----------
    run_names = list(all_runs)
    comparison = {}
    for key in all_runs[run_names[0]]["summary"]:
        row = {"n": all_runs[run_names[0]]["summary"][key]["n"]}
        for name in run_names:
            s = all_runs[name]["summary"].get(key)
            row[name] = s["acc"] if s else None
        comparison[key] = row

    comp_path = out_dir / "comparison.json"
    with open(comp_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    width = max(10, *(len(n) for n in run_names)) + 2
    line = "=" * (34 + width * len(run_names))
    print("\n" + line)
    print(f"{'key':<28} {'n':>5}" + "".join(f"{n:>{width}}" for n in run_names))
    print("-" * (34 + width * len(run_names)))
    for key, v in comparison.items():
        cells = "".join(f"{(v[n] if v[n] is not None else float('nan')):>{width}.4f}" for n in run_names)
        print(f"{key:<28} {v['n']:>5}{cells}")
    print(f"\nSaved: {comp_path}")


if __name__ == "__main__":
    main()