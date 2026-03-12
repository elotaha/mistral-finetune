"""Data preparation: clean, format, and split the dataset.

Usage:
    python src/prepare_data.py
    python src/prepare_data.py --input data/raw.jsonl --train-ratio 0.8 --val-ratio 0.1
"""

import json
import random
import argparse
import logging
from pathlib import Path

# Configure professional logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Project root = parent of src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_jsonl(path: str) -> list[dict]:
    """Load examples from a JSONL file."""
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                examples.append(obj)
            except json.JSONDecodeError as e:
                logging.warning(f"Skipping line {i} due to parse error: {e}")
    return examples


def clean_example(example: dict) -> dict | None:
    """Clean and validate a single example.

    Returns None if the example should be dropped.
    """
    instruction = example.get("instruction", "").strip()
    input_text = example.get("input", "").strip()
    output = example.get("output", "").strip()

    # Drop if missing required fields
    if not instruction or not output:
        return None

    # Drop very short outputs (likely garbage)
    if len(output) < 10:
        return None

    return {
        "instruction": instruction,
        "input": input_text,
        "output": output,
    }


def format_for_training(example: dict) -> str:
    """Format an example into the Mistral [INST] template."""
    instruction = example["instruction"]
    input_text = example["input"]
    output = example["output"]

    if input_text:
        return f"<s>[INST] {instruction}\n\nContext: {input_text} [/INST] {output}</s>"
    else:
        return f"<s>[INST] {instruction} [/INST] {output}</s>"


def save_jsonl(examples: list[dict], path: str):
    """Save examples to a JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Prepare fine-tuning dataset")
    parser.add_argument("--input", type=str, default=str(PROJECT_ROOT / "data" / "raw.jsonl"))
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    test_ratio = round(1.0 - args.train_ratio - args.val_ratio, 2)
    assert test_ratio > 0, f"Train ({args.train_ratio}) + Val ({args.val_ratio}) must be < 1.0"

    logging.info("Starting data preparation pipeline...")

    # 1. Load
    logging.info(f"Loading raw data from: {args.input}")
    raw = load_jsonl(args.input)
    logging.info(f"Loaded {len(raw)} examples.")

    # 2. Clean
    logging.info("Cleaning and validating examples...")
    cleaned = []
    dropped = 0
    for ex in raw:
        result = clean_example(ex)
        if result:
            cleaned.append(result)
        else:
            dropped += 1
    logging.info(f"Kept {len(cleaned)} examples, dropped {dropped}.")

    # 3. Format
    logging.info("Formatting instructions to Mistral [INST] template...")
    for ex in cleaned:
        ex["text"] = format_for_training(ex)

    # 4. Split
    logging.info(f"Splitting dataset: {args.train_ratio:.0%} train / {args.val_ratio:.0%} val / {test_ratio:.0%} test...")
    random.seed(args.seed)
    random.shuffle(cleaned)

    n = len(cleaned)
    train_end = int(n * args.train_ratio)
    val_end = train_end + int(n * args.val_ratio)

    train = cleaned[:train_end]
    val = cleaned[train_end:val_end]
    test = cleaned[val_end:]

    logging.info(f"Split results -> Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

    # 5. Save
    data_dir = Path(args.input).parent
    train_path = data_dir / "train.jsonl"
    val_path = data_dir / "val.jsonl"
    test_path = data_dir / "test.jsonl"

    save_jsonl(train, str(train_path))
    save_jsonl(val, str(val_path))
    save_jsonl(test, str(test_path))

    logging.info("Saved all dataset splits successfully.")

    # 6. Stats
    all_lengths = {
        "Train": [len(ex["text"]) for ex in train],
        "Val": [len(ex["text"]) for ex in val],
        "Test": [len(ex["text"]) for ex in test],
    }
    
    for name, lengths in all_lengths.items():
        avg = sum(lengths) / len(lengths) if lengths else 0
        logging.info(f"Stats [{name}]: avg length = {avg:.0f} chars.")

    every_length = [l for lengths in all_lengths.values() for l in lengths]
    logging.info(f"Maximum example length across all splits: {max(every_length)} chars.")
    logging.info("Data preparation complete.")


if __name__ == "__main__":
    main()

