# -*- coding: utf-8 -*-
# Cleaned from the notebook-exported script (removed Colab/IPython magics).
# Loads filtered_train_simple_models.csv -> builds 10-sample JSONL -> fine-tunes Phi-3.5 Mini briefly.

import json
from pathlib import Path
import pandas as pd
from datasets import load_dataset
from unsloth import FastLanguageModel
from transformers import AutoTokenizer, TrainingArguments, pipeline
from trl import SFTTrainer

SRC = Path("filtered_train_simple_models.csv")   # put this CSV next to the script
OUT_JSONL = Path("train_10_sharegpt.jsonl")
MODEL_NAME = "unsloth/Phi-3.5-mini-Instruct"

def build_jsonl_from_csv(src: Path, out_path: Path, n_samples: int = 10, seed: int = 42):
    df = pd.read_csv(src)
    df = df.dropna(subset=["annotation", "json_desc"])
    sampled = df.sample(n=n_samples, random_state=seed).copy()
    sampled["json_desc_sanitized"] = sampled["json_desc"].astype(str).str.replace(",", " <COMMA> ")

    with out_path.open("w", encoding="utf-8") as f:
        for _, row in sampled.iterrows():
            rec = {
                "uid": row.get("uid", ""),
                "conversations": [
                    {"from": "human", "value": str(row["annotation"])},
                    {"from": "gpt", "value": str(row["json_desc_sanitized"])},
                ],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"✅ Wrote {len(sampled)} examples to {out_path.resolve()}")

def to_messages(example):
    msgs = []
    for turn in example["conversations"]:
        role = "user" if turn["from"] == "human" else "assistant"
        msgs.append({"role": role, "content": turn["value"]})
    return {"messages": msgs}

def main():
    # 1) Make the tiny train set
    if not SRC.exists():
        raise FileNotFoundError(f"Missing {SRC}. Put filtered_train_simple_models.csv next to this script.")
    build_jsonl_from_csv(SRC, OUT_JSONL, n_samples=10, seed=42)

    # 2) Load dataset and map to chat messages
    raw_ds = load_dataset("json", data_files=str(OUT_JSONL), split="train")
    chat_ds = raw_ds.map(to_messages, remove_columns=[c for c in raw_ds.column_names if c != "uid"]).shuffle(seed=42)

    # 3) Load model (4-bit) + LoRA
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        load_in_4bit = True,
        dtype = None,
        device_map = "auto",
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        lora_alpha = 16,
        lora_dropout = 0.05,
        target_modules = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    )
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token

    # 4) Tokenize with chat template
    def format_example(example):
        return tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
    def tokenize_fn(example):
        text = format_example(example)
        return tokenizer(
            text, max_length=2048, truncation=True, padding="max_length", return_tensors="pt"
        )
    tokenized = chat_ds.map(tokenize_fn, batched=False, remove_columns=chat_ds.column_names)

    # 5) Tiny SFT
    trainer = SFTTrainer(
        model=model,
        train_dataset=tokenized,
        tokenizer=tokenizer,
        max_seq_length=2048,
        dataset_text_field=None,  # already tokenized
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            num_train_epochs=1,
            learning_rate=2e-5,
            fp16=True,
            logging_steps=1,
            output_dir="phi35-mini-sft",
            save_strategy="no",
            report_to="none",
        ),
    )
    trainer.train()
    trainer.save_model("phi35-mini-sft")
    tokenizer.save_pretrained("phi35-mini-sft")
    print("✅ Training done and model saved to ./phi35-mini-sft")

    # 6) Quick test
    gen = pipeline("text-generation", model="phi35-mini-sft", tokenizer=tokenizer, device_map="auto")
    prompt = "Given a CAD specification, output JSON-like CAD steps with <COMMA> instead of commas."
    out = gen(prompt, max_new_tokens=200, do_sample=True, temperature=0.7)
    print(out[0]["generated_text"])

if __name__ == "__main__":
    main()
