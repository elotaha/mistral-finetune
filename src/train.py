"""QLoRA fine-tuning of Mistral 7B Instruct.

Usage:
    python src/train.py
    python src/train.py --epochs 5 --lr 1e-4
"""

import os
import json
import argparse
import logging

# Set MLflow experiment name before initializing
os.environ["MLFLOW_EXPERIMENT_NAME"] = "mistral-qlora-finetuning"

# Configure professional logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer


MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
OUTPUT_DIR = "outputs"


def load_jsonl(path: str) -> list[dict]:
    """Load a JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Mistral 7B with QLoRA")
    parser.add_argument("--train", type=str, default="data/train.jsonl")
    parser.add_argument("--val", type=str, default="data/val.jsonl")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    args = parser.parse_args()

    print("=" * 50)
    print("🧠 QLoRA FINE-TUNING")
    print(f"   Model: {MODEL_NAME}")
    print(f"   Epochs: {args.epochs}")
    print(f"   LR: {args.lr}")
    print(f"   LoRA r={args.lora_r}, alpha={args.lora_alpha}")
    print("=" * 50)

    # --- Load data ---
    print("\n📂 Loading data...")
    train_data = Dataset.from_list(load_jsonl(args.train))
    val_data = Dataset.from_list(load_jsonl(args.val))
    print(f"   Train: {len(train_data)}  Val: {len(val_data)}")

    # --- Load tokenizer ---
    print("\n⏳ Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Load model (4-bit quantized) ---
    print("⏳ Loading model (4-bit)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    # --- LoRA config ---
    print("🔧 Applying LoRA...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)

    trainable, total = model.get_nb_trainable_parameters()
    print(f"   Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # --- Training ---
    print("\n🚀 Starting training...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_steps=2,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        fp16=False,
        report_to="mlflow",
        optim="paged_adamw_8bit",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=training_args,
    )

    trainer.train()

    # --- Save ---
    final_path = f"{OUTPUT_DIR}/final"
    print(f"\n💾 Saving adapter to {final_path}")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)

    print("\n✅ Fine-tuning complete.")
    print(f"   Adapter: {final_path}")
    print(f"   Use with: python src/inference.py --adapter {final_path}")


if __name__ == "__main__":
    main()
