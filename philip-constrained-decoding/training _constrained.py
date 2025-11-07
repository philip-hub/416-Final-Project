# --- johnstraincode_constrained_v2.py ---
# Adds schema-constrained decoding (Outlines) and FIXES dataset target mapping to `json_desc`.
# Includes filtering for rows missing `json_desc`, and a tiny eval after training.
# Original base: johnstraincode.py

import json
import os
import sys
import warnings
from typing import List

import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

warnings.filterwarnings("ignore")

# ----------------------------------------
# Optional deps for constrained decoding
# ----------------------------------------
try:
    import outlines
    from outlines.models.transformers import Transformer as OutlinesTransformer
    from outlines.generate.json import json as outlines_json
except Exception:
    outlines = None

try:
    from jsonschema import Draft7Validator
except Exception:
    Draft7Validator = None

try:
    from json_repair import repair_json
except Exception:
    repair_json = None

# ----------------------------------------
# bitsandbytes detection (QLoRA or fallback LoRA)
# ----------------------------------------
use_quantization = False
try:
    import bitsandbytes as bnb
    _ = bnb.nn.Linear4bit(10, 10)  # sanity test
    use_quantization = True
    print("✅ bitsandbytes loaded successfully - QLoRA (4-bit) will be used")
except Exception as e:
    use_quantization = False
    print(f"⚠️  bitsandbytes issue detected: {type(e).__name__}")
    print("   ✅ Fallback: Standard LoRA (no quantization)")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("=" * 70)
print("🎯 TRAINING MODE DETECTED")
print("=" * 70)
print("✅ Mode:", "QLoRA (4-bit Quantization)" if use_quantization else "Standard LoRA (No Quantization)")
if torch.cuda.is_available():
    print(f"\n📍 Device: {device}")
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
print("=" * 70)

# ----------------------------------------
# Base model / tokenizer
# ----------------------------------------
torch.manual_seed(42)
model_name = "microsoft/Phi-3-mini-128k-instruct"

print(f"Using device: {device}")
print(f"CUDA available: {torch.cuda.is_available()}")

print(f"Loading tokenizer from {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    padding_side="right",
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
print("✅ Tokenizer loaded")

# ----------------------------------------
# Load CADmium dataset
# Expect columns: ['uid', 'annotation', 'json_desc']
# ----------------------------------------
print("Loading CADmium dataset...")
dataset = load_dataset("chandar-lab/CADmium-ds", split="train", streaming=False)

print(f"Dataset columns: {dataset.column_names}")
if not all(k in dataset.column_names for k in ["annotation", "json_desc"]):
    raise RuntimeError("Expected columns 'annotation' and 'json_desc' not found in dataset.")

# Keep a manageable subset for demo; increase as VRAM allows
subset_n = min(2000, len(dataset))  # adjust as needed
dataset = dataset.shuffle(seed=42).select(range(subset_n))

# Filter out rows without json_desc
def _has_json_desc(ex):
    jd = ex.get("json_desc")
    return isinstance(jd, (str, dict)) and len(str(jd).strip()) > 0

dataset = dataset.filter(_has_json_desc)
print(f"✅ Loaded & filtered: {len(dataset)} examples")

# ----------------------------------------
# JSON Schema (liberal on sketch, strict on outer shape)
# ----------------------------------------
CADMIUM_JSON_SCHEMA = {
    "type": "object",
    "required": ["parts"],
    "additionalProperties": False,
    "properties": {
        "parts": {
            "type": "object",
            "additionalProperties": False,
            "patternProperties": {
                r"^part_\d+$": {
                    "type": "object",
                    "additionalProperties": True,
                    "required": ["coordinate_system", "sketch", "extrusion"],
                    "properties": {
                        "coordinate_system": {
                            "type": "object",
                            "additionalProperties": True,
                            "required": ["Euler Angles", "Translation Vector"],
                            "properties": {
                                "Euler Angles": {
                                    "type": "array",
                                    "minItems": 3,
                                    "maxItems": 3,
                                    "items": {"type": "number"}
                                },
                                "Translation Vector": {
                                    "type": "array",
                                    "minItems": 3,
                                    "maxItems": 3,
                                    "items": {"type": "number"}
                                }
                            }
                        },
                        "sketch": {
                            "type": "object",
                            "additionalProperties": True
                        },
                        "extrusion": {
                            "type": "object",
                            "additionalProperties": True,
                            "required": [
                                "extrude_depth_towards_normal",
                                "extrude_depth_opposite_normal",
                                "sketch_scale",
                                "operation"
                            ],
                            "properties": {
                                "extrude_depth_towards_normal": {"type": "number"},
                                "extrude_depth_opposite_normal": {"type": "number"},
                                "sketch_scale": {"type": "number"},
                                "operation": {"type": "string"}
                            }
                        },
                        "description": {
                            "type": "object",
                            "additionalProperties": True,
                            "properties": {
                                "name": {"type": "string"},
                                "shape": {"type": "string"},
                                "length": {"type": "number"},
                                "width": {"type": "number"},
                                "height": {"type": "number"}
                            }
                        }
                    }
                }
            }
        }
    }
}

# ----------------------------------------
# Format dataset to chat-style messages with correct target mapping
# ----------------------------------------
def format_cad_instruction(example):
    instruction = (example.get("annotation") or "").strip()
    raw_json = example.get("json_desc")

    # normalize target JSON
    cleaned = "{}"
    if isinstance(raw_json, dict):
        cleaned = json.dumps(raw_json, separators=(",", ":"), ensure_ascii=False)
    elif isinstance(raw_json, str) and raw_json.strip():
        try:
            obj = json.loads(raw_json)
            cleaned = json.dumps(obj, separators=(",", ":"), ensure_ascii=False)
        except Exception:
            # keep raw if it's nearly JSON; inference repair will help
            cleaned = raw_json.strip()

    # ensure it at least starts like a JSON object
    if not cleaned.startswith("{"):
        cleaned = "{}"

    messages = [
        {
            "role": "system",
            "content": "You map natural language instructions to a corresponding CADmium JSON. Return ONLY valid JSON (no prose). Units: meters."
        },
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": cleaned},
    ]
    return {"messages": messages}

print("Formatting dataset...")
formatted_dataset = dataset.map(
    format_cad_instruction,
    remove_columns=dataset.column_names,
    desc="Formatting CAD instructions"
)
print(f"✅ Formatted {len(formatted_dataset)} examples")
print("Example messages:", formatted_dataset[0]["messages"][:2], "...")

# ----------------------------------------
# Model loading + LoRA
# ----------------------------------------
if use_quantization:
    # choose compute dtype
    if torch.cuda.is_available():
        try:
            _ = torch.zeros(1, dtype=torch.bfloat16, device="cuda")
            quant_dtype = torch.bfloat16
        except Exception:
            quant_dtype = torch.float16
    else:
        quant_dtype = torch.float32

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=quant_dtype,
        bnb_4bit_use_double_quant=True,
    )
    print("✅ QLoRA: 4-bit quantization config ready")
else:
    bnb_config = None

print(f"Loading model {model_name}...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    attn_implementation="eager",
    quantization_config=bnb_config,
)

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
)

model = get_peft_model(model, lora_config)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"✅ LoRA ready. Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

# ----------------------------------------
# Training args + SFTTrainer
# ----------------------------------------
output_dir = "./phi3-cad-lora"

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=2,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=5,
    save_strategy="epoch",
    save_total_limit=2,
    fp16=False,
    bf16=True,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit" if use_quantization else "adamw_torch",
    report_to="none",
    push_to_hub=False,
)

def formatting_prompts_func(examples):
    texts = []
    for messages in examples["messages"]:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(text)
    return texts

trainer = SFTTrainer(
    model=model,
    train_dataset=formatted_dataset,
    args=training_args,
    formatting_func=formatting_prompts_func,
)

print("🚀 Starting training...")
trainer.train()
print("✅ Training complete.")

# Save adapters
adapter_dir = "./phi3-cad-lora-adapters"
model.save_pretrained(adapter_dir)
tokenizer.save_pretrained(adapter_dir)
print(f"✅ LoRA adapters saved to: {adapter_dir}")

# ----------------------------------------
# Inference utilities (constrained + fallback)
# ----------------------------------------
def _base_prompt(instruction: str) -> str:
    messages = [
        {"role": "system", "content": (
            "You map natural language instructions to a corresponding CADmium JSON. "
            "Return ONLY valid JSON (no prose). Units in meters where applicable."
        )},
        {"role": "user", "content": instruction},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def generate_cad_json_unconstrained(instruction: str, max_new_tokens=512, temperature=0.7, top_p=0.9) -> str:
    prompt = _base_prompt(instruction)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=False,
        )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "<|assistant|>" in text:
        text = text.split("<|assistant|>")[-1].strip()
    return text

def generate_cad_json_constrained(instruction: str, max_new_tokens=512, temperature=0.7, top_p=0.9) -> str:
    prompt = _base_prompt(instruction)
    if outlines is not None:
        print("🔒 Using Outlines constrained decoding")
        try:
            outlines_model = OutlinesTransformer(model=model, tokenizer=tokenizer)
            generator = outlines_json(CADMIUM_JSON_SCHEMA)
            text = generator(
                outlines_model,
                prompt,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        except Exception as e:
            print(f"⚠️ Outlines decoding failed ({type(e).__name__}). Falling back.")
            text = generate_cad_json_unconstrained(instruction, max_new_tokens, temperature, top_p)
    else:
        print("🟡 Outlines not available — using fallback generation")
        text = generate_cad_json_unconstrained(instruction, max_new_tokens, temperature, top_p)

    # Optional repair & validation
    if repair_json is not None:
        try:
            text = repair_json(text)
        except Exception:
            pass

    if Draft7Validator is not None:
        try:
            obj = json.loads(text)
            validator = Draft7Validator(CADMIUM_JSON_SCHEMA)
            errs = list(validator.iter_errors(obj))
            if errs:
                print(f"⚠️ JSON does not fully match schema ({len(errs)} issue(s)).")
        except Exception:
            print("⚠️ JSON parse failed post-generation.")
    return text

# ----------------------------------------
# Tiny post-training evaluation
# ----------------------------------------
eval_prompts: List[str] = [
    "Create a rectangular sketch 10mm by 20mm centered at the origin",
    "Make a circular sketch with radius 5mm at position (10, 10)",
    "Create a cube with side length 10 mm",
    "Design a cylinder with radius 3mm and height 15mm"
]

print("\n🧪 Post-training eval with constrained decoding")
print("=" * 80)
for i, p in enumerate(eval_prompts, 1):
    print(f"\n📝 Prompt {i}: {p}")
    print("-" * 80)
    out = generate_cad_json_constrained(p, max_new_tokens=256)
    print(out)
    print("-" * 80)
print("=" * 80)
print("✅ Eval finished")
