import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline 
from peft import PeftModel

base_model_name = "microsoft/Phi-3-mini-128k-instruct"
adapter_path = "./checkpoint-1701"

tokenizer = AutoTokenizer.from_pretrained(
    base_model_name,
    trust_remote_code=True,
    padding_side="right"
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",
)
model = PeftModel.from_pretrained(
    base_model,
    adapter_path,
    device_map="auto"
)

SYSTEM_PROMPT = (
    "You are a CAD parameter predictor. Given a natural language instruction, "
    "predict the appropriate CAD model structure in JSON format. "
    "Output hierarchical JSON with parts, coordinate systems, descriptions, and sketches. "
    "Use the minimal JSON format: only include relevant parameters for the instruction. "
    "Omit parameters that are not mentioned or irrelevant to the design."
)

def generate_cad_json(instruction, max_new_tokens=2048, do_sample=False, temperature=None, top_p=None):
    """
    Generate CAD JSON from natural language instruction.
    
    Args:
        instruction: Natural language CAD instruction
        max_new_tokens: Maximum tokens to generate (default: 2048)
        do_sample: Use sampling (False=greedy, True=stochastic). Default False for deterministic CAD generation
        temperature: Sampling temperature (only used if do_sample=True). Lower=more focused, higher=more random
        top_p: Nucleus sampling threshold (only used if do_sample=True)
    
    Recommended settings:
        - For deterministic CAD JSON: do_sample=False (default)
        - For creative variations: do_sample=True, temperature=0.3-0.5, top_p=0.9
    """
    
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT  # Use the same prompt as training
        },
        {
            "role": "user",
            "content": instruction
        }
    ]
    
    # Format using chat template
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    input_length = inputs.input_ids.shape[1]
    
    # Build generation config
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": False,  # Disable cache to avoid DynamicCache error
    }
    
    # Add sampling parameters only if do_sample=True
    if do_sample:
        if temperature is not None:
            gen_kwargs["temperature"] = temperature
        if top_p is not None:
            gen_kwargs["top_p"] = top_p
    
    # Generate WITHOUT cache (slower but avoids compatibility issues)
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
    
    # Decode only the generated part (skip input prompt)
    generated_ids = outputs[0][input_length:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return generated_text

def generate_model(prompt):
    return generate_cad_json(prompt, max_new_tokens=2048)