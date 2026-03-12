"""Interactive inference with the fine-tuned model.

Usage:
    python src/inference.py
    python src/inference.py --adapter outputs/final
    python src/inference.py --base-only
"""

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


MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"


def load_model(adapter_path: str | None = None):
    """Load base model, optionally with LoRA adapter."""
    logging.info(f"Loading base model {MODEL_NAME}...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    if adapter_path:
        logging.info(f"Loading adapter weights from {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    label = "fine-tuned" if adapter_path else "base"
    logging.info(f"Model loaded successfully ({label} mode).")
    return model, tokenizer


def generate(model, tokenizer, question: str, max_new_tokens: int = 300) -> str:
    """Generate an answer."""
    prompt = f"<s>[INST] {question} [/INST]"
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


def main():
    parser = argparse.ArgumentParser(description="Interactive inference")
    parser.add_argument("--adapter", type=str, default="outputs/final")
    parser.add_argument("--base-only", action="store_true", help="Use base model only")
    args = parser.parse_args()

    adapter = None if args.base_only else args.adapter
    model, tokenizer = load_model(adapter)

    print("=" * 50)
    print("INTERACTIVE INFERENCE SESSION")
    print("Type your question, or 'quit' to exit.")
    print("=" * 50)

    while True:
        try:
            question = input("\nUser > ").strip()
            if question.lower() in ("quit", "exit", "q"):
                print("Session terminated.")
                break
            if not question:
                continue

            answer = generate(model, tokenizer, question)
            print(f"\nAssistant > {answer}")
        
        except KeyboardInterrupt:
            print("\nSession terminated.")
            break


if __name__ == "__main__":
    main()
