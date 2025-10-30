# -*- coding: utf-8 -*-
# Cleaned from the notebook-exported script (removed Colab/IPython magics).

import json
from pathlib import Path
import pandas as pd
from datasets import load_dataset
from unsloth import FastLanguageModel
from transformers import AutoTokenizer, TrainingArguments, pipeline
from trl import SFTTrainer

SRC = Path("filtered_train_simple_models.csv")
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
    if not SRC.exists():
        raise FileNotFoundError(f"Missing {SRC}.")
    build_jsonl_from_csv(SRC, OUT_JSONL, n_samples=10, seed=42)

    raw_ds = load_dataset("json", data_files=str(OUT_JSONL), split="train")
    chat_ds = raw_ds.map(to_messages, remove_columns=[c for c in raw_ds.column_names if c != "uid"]).shuffle(seed=42)

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

    trainer = SFTTrainer(
        model=model,
        train_dataset=tokenized,
        tokenizer=tokenizer,
        max_seq_length=2048,
        dataset_text_field=None,
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

    gen = pipeline("text-generation", model="phi35-mini-sft", tokenizer=tokenizer, device_map="auto")
    print(gen("Say hi to Philip.", max_new_tokens=40)[0]["generated_text"])

if __name__ == "__main__":
    main()
