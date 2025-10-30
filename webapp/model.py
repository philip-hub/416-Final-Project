import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline 

torch.manual_seed(0)

model_name = "microsoft/Phi-3-mini-4k-instruct"

# Try to load on GPU, fall back to CPU if needed
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map=device,
    torch_dtype="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct", trust_remote_code=True)

def generate_model(prompt):
    messages = f"""<|system|>
    You map natural language instructions to a corresponding Fusion 360 json
    <|end|><|user|>
    {prompt}
    <|end|><|assistant|>"""

    # Tokenize the message
    inputs = tokenizer(messages, return_tensors="pt").to(device)

    # Generate with use_cache=False to avoid compatibility issues
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            use_cache=False,
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)