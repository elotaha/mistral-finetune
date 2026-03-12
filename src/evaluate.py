"""Evaluate base model vs fine-tuned model on the validation set.

Computes ROUGE-L and BERTScore, prints a comparison table.

Usage:
    python src/evaluate.py
    python src/evaluate.py --adapter outputs/final --samples 20
"""

import json
import argparse
import logging

# Configure professional logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from rouge_score import rouge_scorer
from bert_score import score as bert_score


MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"


def load_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_prompt(example: dict) -> str:
    """Build the prompt (without the answer) for generation."""
    instruction = example["instruction"]
    input_text = example.get("input", "")
    if input_text:
        return f"<s>[INST] {instruction}\n\nContext: {input_text} [/INST]"
    return f"<s>[INST] {instruction} [/INST]"


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    """Generate a response from the model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def compute_metrics(predictions: list[str], references: list[str]) -> dict:
    """Compute ROUGE-L and BERTScore."""
    # ROUGE-L
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_scores = [scorer.score(ref, pred)["rougeL"].fmeasure
                    for pred, ref in zip(predictions, references)]
    avg_rouge = sum(rouge_scores) / len(rouge_scores)

    # BERTScore
    P, R, F1 = bert_score(predictions, references, lang="en", verbose=False)
    avg_bert = F1.mean().item()

    return {"rouge_l": avg_rouge, "bertscore_f1": avg_bert}


def main():
    parser = argparse.ArgumentParser(description="Evaluate base vs fine-tuned model")
    parser.add_argument("--val", type=str, default="data/val.jsonl")
    parser.add_argument("--adapter", type=str, default="outputs/final")
    parser.add_argument("--samples", type=int, default=20, help="Number of val samples")
    args = parser.parse_args()

    logging.info("Starting model evaluation pipeline...")

    # Load val data
    val_data = load_jsonl(args.val)[:args.samples]
    references = [ex["output"] for ex in val_data]
    prompts = [build_prompt(ex) for ex in val_data]

    logging.info(f"Evaluating on {len(val_data)} validation examples.")

    # Load base model
    logging.info("Loading base model (4-bit quantized)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    base_model.eval()

    # Generate with base model
    logging.info("Generating responses with base model...")
    base_preds = [generate(base_model, tokenizer, p) for p in prompts]

    # Load fine-tuned adapter
    logging.info(f"Loading fine-tuned LoRA adapter from {args.adapter}...")
    ft_model = PeftModel.from_pretrained(base_model, args.adapter)
    ft_model.eval()

    # Generate with fine-tuned model
    logging.info("Generating responses with fine-tuned model...")
    ft_preds = [generate(ft_model, tokenizer, p) for p in prompts]

    # Compute metrics
    logging.info("Computing evaluation metrics (ROUGE-L, BERTScore)...")
    base_metrics = compute_metrics(base_preds, references)
    ft_metrics = compute_metrics(ft_preds, references)

    # Print results
    print("\n" + "=" * 50)
    print("RESULTS COMPARISON")
    print("=" * 50)
    print(f"{'Metric':<20} {'Base':>10} {'Fine-Tuned':>12} {'Delta':>8}")
    print("-" * 50)

    for metric in ["rouge_l", "bertscore_f1"]:
        base_val = base_metrics[metric]
        ft_val = ft_metrics[metric]
        delta = ft_val - base_val
        sign = "+" if delta > 0 else ""
        print(f"{metric:<20} {base_val:>10.4f} {ft_val:>12.4f} {sign}{delta:>7.4f}")

    # Show example comparisons
    print("\n" + "=" * 50)
    print("QUALITATIVE EXAMPLES (First 3)")
    print("=" * 50)
    for i in range(min(3, len(val_data))):
        print(f"\n--- Example {i+1} ---")
        print(f"Q: {val_data[i]['instruction']}")
        print(f"Expected: {references[i][:150]}...")
        print(f"Base:     {base_preds[i][:150]}...")
        print(f"FT:       {ft_preds[i][:150]}...")

    logging.info("Evaluation pipeline completed.")


if __name__ == "__main__":
    main()
