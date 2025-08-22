#!/usr/bin/env python3
"""
Baseline (Pre-Fine-Tuning) Evaluator

Runs the base generative model (no LoRA) on N questions and saves a CSV with
question, expected, answer, response time, similarity score, and correctness.

Usage:
  python tests/baseline_eval.py --dataset data/test/comprehensive_qa.json --num 10 \
      --out results
"""
from __future__ import annotations

import os
import sys
import time
import json
import random
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def jaccard_similarity(a: str, b: str) -> float:
    a_words = set((a or "").lower().split())
    b_words = set((b or "").lower().split())
    if not a_words or not b_words:
        return 0.0
    inter = len(a_words & b_words)
    union = len(a_words | b_words)
    return inter / union if union else 0.0


def simple_confidence(answer: str) -> float:
    conf = 0.75
    al = (answer or "").lower()
    if any(p in al for p in ["don't know", "not sure", "unknown", "not provided"]):
        conf -= 0.2
    wc = len(al.split())
    if wc < 5:
        conf -= 0.1
    if any(ch.isdigit() for ch in al):
        conf += 0.05
    return max(0.5, min(0.9, conf))


def load_base_model(model_path_local: str = "models/Llama-3.1-8B-Instruct",
                    model_repo: str = "meta-llama/Llama-3.1-8B-Instruct"):
    # Prefer local model if present, else fallback to HF Hub
    path = model_path_local if os.path.isdir(model_path_local) else model_repo
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        path,
        device_map="auto" if device == "cuda" else None,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        low_cpu_mem_usage=True
    )

    return tokenizer, model, device


from typing import Tuple


def generate_answer(tokenizer, model, device: str, question: str,
                    max_new_tokens: int = 128, temperature: float = 0.2) -> Tuple[str, float]:
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful AI assistant specialized in financial analysis. Answer questions accurately based on financial data.<|eot_id|><|start_header_id|>user<|end_header_id|>
{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    if device == "cuda":
        inputs = {k: v.to(device) for k, v in inputs.items()}

    start = time.time()
    with torch.no_grad():
        # Use deterministic greedy decoding for baseline; avoid unsupported flags warnings
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    elapsed = time.time() - start

    input_len = inputs['input_ids'].shape[1]
    gen_tokens = outputs[0][input_len:]
    answer = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    if "<|endoftext|>" in answer:
        answer = answer.split("<|endoftext|>")[0].strip()
    return answer, elapsed


def evaluate(dataset_path: str, num_questions: int, out_dir: str,
             model_path: str = "models/Llama-3.1-8B-Instruct",
             filename_prefix: str = "baseline_eval") -> Path:
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Normalize to list of dicts with keys instruction/output
    items: List[Dict] = []
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict) and 'financial_qa_pairs' in data:
        # Training dataset format
        items = data['financial_qa_pairs']
    else:
        raise ValueError("Unsupported dataset format for baseline eval")

    random.seed(42)
    sample = random.sample(items, min(num_questions, len(items)))

    tokenizer, model, device = load_base_model(model_path_local=model_path)

    rows = []
    correct = 0
    for i, qa in enumerate(sample, start=1):
        question = qa.get('instruction') or qa.get('question') or ""
        expected = qa.get('output') or qa.get('answer') or ""
        ans, t = generate_answer(tokenizer, model, device, question)
        sim = jaccard_similarity(ans, expected)
        conf = simple_confidence(ans)
        is_correct = sim > 0.6
        correct += 1 if is_correct else 0
        rows.append({
            'id': i,
            'question': question,
            'expected': expected,
            'answer': ans,
            'similarity': round(sim, 4),
            'confidence': round(conf, 4),
            'time_s': round(t, 4),
            'correct': int(is_correct),
            'model': model_path
        })

    df = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(out_dir) / f"{filename_prefix}_{ts}.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")

    acc = correct / len(rows) if rows else 0.0
    avg_time = sum(r['time_s'] for r in rows) / len(rows) if rows else 0.0
    # Avoid non-ASCII symbols for Windows console compatibility
    print(f"[OK] Saved: {out_path}")
    print(f"Accuracy: {acc:.2%} | Avg time: {avg_time:.2f}s | Questions: {len(rows)}")
    return out_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Baseline pre-FT evaluator")
    parser.add_argument('--dataset', type=str, default='data/test/comprehensive_qa.json')
    parser.add_argument('--num', type=int, default=10)
    parser.add_argument('--out', type=str, default='results')
    parser.add_argument('--model', type=str, default='models/Llama-3.1-8B-Instruct', help='Path or repo of the model to evaluate')
    parser.add_argument('--prefix', type=str, default='baseline_eval', help='Output CSV filename prefix')
    args = parser.parse_args(argv)

    try:
        evaluate(args.dataset, args.num, args.out, model_path=args.model, filename_prefix=args.prefix)
        return 0
    except Exception as e:
        print(f"[X] Baseline evaluation failed: {e}")
        return 1


if __name__ == '__main__':
    raise SystemExit(main())


