# LoRA Fine-tuning Demo: Phi-3-mini for CAD Generation

A complete implementation of LoRA (Low-Rank Adaptation) fine-tuning for Phi-3-mini-128k to create a Natural Language → CAD JSON generation model using the CADmium dataset.

## 📋 Overview

This demo implements the recommended architecture and best practices for training a CAD-to-Language model:

- **Base Model**: `microsoft/Phi-3-mini-128k-instruct` (longer context for complex CAD designs)
- **Training Method**: QLoRA (4-bit quantization + LoRA adapters)
- **Dataset**: CADmium dataset from Hugging Face (`chandar-lab/CADmium`)
- **Task**: Natural Language → Fusion 360 CAD JSON generation

## 🎯 Key Features

### ✅ Implemented Guardrails (Day One)

1. **Constrained Decoding**: JSON-Schema validation during generation
2. **Canonical Operation Order**: Enforces `sketch → feature → transform` ordering
3. **Unit & Frame Validation**: Consistent units (meters) and frame declarations
4. **Deterministic Numeric Formatting**: Rounds to 1e-6, forbids scientific notation
5. **Validation Metrics**: Schema-valid rate, Structural F1, Geometric accuracy

### 🔧 Technical Configuration

```python
# QLoRA Configuration
- Quantization: 4-bit NF4 with double quantization
- Compute dtype: bfloat16
- LoRA rank: 16, alpha: 16, dropout: 0.05
- Target modules: All attention (q/k/v/o) + MLP (gate/up/down)

# Training Configuration
- Learning rate: 2e-4 with cosine schedule
- Warmup: 3% of total steps
- Sequence length: 2048 tokens (scalable to 4k)
- Effective batch size: 8 (1 per device × 8 accumulation)
- Epochs: 2-3 with early stopping
- Optimizer: 8-bit paged AdamW
```

## 📁 Project Structure

```
Demo/
├── LoRADemo.ipynb              # Main training notebook
├── cad_validators.py           # Comprehensive validation system
├── constrained_decoding.py     # Schema-constrained generation
├── CAD_Training_Guide.md       # Detailed training guide
└── README_LoRA.md             # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install transformers datasets peft accelerate bitsandbytes trl torch sentencepiece
```

### 2. Run the Notebook

Open `LoRADemo.ipynb` and run cells sequentially:

1. **Cell 1-2**: Import libraries and configure environment
2. **Cell 3-4**: Load and prepare CADmium dataset (subset)
3. **Cell 5-7**: Configure QLoRA and LoRA adapters
4. **Cell 8-9**: Load base model with 4-bit quantization
5. **Cell 10-11**: Configure and initialize SFT trainer
6. **Cell 12**: Train the model (this will take time!)
7. **Cell 13**: Save LoRA adapters
8. **Cell 14-15**: Test generation with sample prompts
9. **Cell 16-17**: Validate outputs with JSON schema

### 3. Customize for Your Needs

```python
# Adjust dataset size (in LoRADemo.ipynb)
num_samples = 100  # Increase for better results (500-5000 recommended)

# Adjust training epochs
num_train_epochs = 2  # 2-3 epochs recommended

# Adjust LoRA rank for model capacity
r = 16  # 16-32 recommended
```

## 📊 Expected Results

### Training Metrics (100 samples, 2 epochs)
- **Training Time**: ~5-15 minutes (depends on GPU)
- **Memory Usage**: ~8-12 GB GPU memory
- **Trainable Parameters**: ~0.5-2% of total parameters

### Validation Targets (Production)
- **Schema-valid rate**: ≥99% (target)
- **Structural F1**: ≥97% (target)
- **Geometric accuracy**: ≥97% (target)

*Note: Demo with 100 samples won't reach production targets. Scale up to 500-5000 samples.*

## 🔬 Validation System

### Using the Validator

```python
from cad_validators import CADValidator, format_validation_report

# Initialize validator
validator = CADValidator(unit_system="meters", precision=1e-6)

# Validate generated JSON
is_valid, issues = validator.validate(generated_json)
print(format_validation_report(is_valid, issues))

# Calculate batch metrics
metrics = validator.calculate_metrics(predictions, references)
print(f"Schema valid rate: {metrics['schema_valid_rate']:.2%}")
print(f"Structural F1: {metrics['structural_f1']:.2%}")
```

### Validation Categories

1. **Syntax**: Valid JSON parsing
2. **Schema**: Required keys and structure
3. **Units**: Consistent unit declarations
4. **Operation Order**: Canonical ordering (sketch → feature → transform)
5. **Numerics**: Precision and format validation
6. **Frames**: Valid coordinate frame declarations

## 🎨 Example Usage

### Generation

```python
from constrained_decoding import CADJSONGenerator, normalize_cad_json

# Initialize generator
generator = CADJSONGenerator(model, tokenizer, validator)

# Generate with validation and retries
result = generator.generate(
    instruction="Create a cylinder with radius 5mm and height 20mm",
    max_retries=3
)

print(f"Valid: {result['is_valid']}")
print(f"Attempts: {result['attempts']}")
print(f"JSON:\n{result['json']}")
```

### Input/Output Example

**Input:**
```
Create a rectangular sketch 10mm by 20mm centered at the origin, 
then extrude it 5mm upward.
```

**Output:**
```json
{
  "parts": {
    "part_0": {
      "frame": "world",
      "units": "meters",
      "sketch": {
        "type": "rectangle",
        "center": [0.0, 0.0],
        "width": 0.01,
        "height": 0.02
      },
      "extrusion": {
        "distance": 0.005,
        "operation": "NewBodyFeatureOperation"
      }
    }
  }
}
```

## 🔄 Advanced: Dual Adapter Setup

For production, train two LoRA adapters on the same base:

### Adapter 1: NL → CAD-JSON (Generation)
```python
# Train with instruction → CAD JSON pairs
# This is what the current notebook does
```

### Adapter 2: CAD-JSON → NL (Explanation)
```python
# Train with CAD JSON → description pairs
# Reverse the data format in preprocessing
```

### Hot-Swap at Inference
```python
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-128k-instruct",
    quantization_config=bnb_config,
    device_map="auto"
)

# Load both adapters
model_gen = PeftModel.from_pretrained(base_model, "./phi3-cad-lora-adapters")
model_exp = PeftModel.from_pretrained(base_model, "./phi3-cad-explain-adapters")

# Switch adapters
# model.set_adapter("generation")  # For NL → CAD
# model.set_adapter("explanation") # For CAD → NL
```

## 📈 Scaling Up

### For Production Deployment

1. **Increase Dataset Size**
   - Use 500-5000 samples minimum
   - Include diverse CAD operations
   - Add hard negatives (wrong units, wrong order, etc.)

2. **Add Synthetic Data**
   - Parameter sweeps (radii, centers, depths, rotations)
   - Systematic operation combinations
   - Edge cases and error corrections

3. **Improve Training**
   - Train for 3-5 epochs with early stopping
   - Use validation set for monitoring
   - Implement learning rate finder
   - Add gradient clipping

4. **Enhanced Validation**
   - Integrate with Fusion 360 API for real execution tests
   - Add geometric constraint verification
   - Implement parametric validity checks
   - Test assembly compatibility

5. **Production Deployment**
   - Add request caching
   - Implement streaming generation
   - Add user feedback loop
   - Monitor metrics in production

## 🐛 Troubleshooting

### Out of Memory (OOM)
```python
# Reduce batch size
per_device_train_batch_size = 1

# Reduce sequence length
max_seq_length = 1024

# Reduce dataset size
num_samples = 50

# Enable gradient checkpointing (already enabled)
gradient_checkpointing = True
```

### Slow Training
```python
# Use smaller dataset for quick iteration
num_samples = 20

# Reduce epochs
num_train_epochs = 1

# Check GPU utilization
nvidia-smi
```

### Poor Generation Quality
```python
# Increase dataset size (most common issue)
num_samples = 500  # or more

# Train longer
num_train_epochs = 3

# Adjust generation parameters
temperature = 0.5  # Lower for more deterministic
top_p = 0.95       # Adjust nucleus sampling
```

## 📚 Resources

- **Paper**: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- **Phi-3**: [Technical Report](https://arxiv.org/abs/2404.14219)
- **PEFT**: [Hugging Face PEFT Documentation](https://huggingface.co/docs/peft)
- **TRL**: [Transformer Reinforcement Learning](https://huggingface.co/docs/trl)
- **CADmium**: [Dataset on Hugging Face](https://huggingface.co/datasets/chandar-lab/CADmium)

## 🤝 Contributing

This is a demo/educational project. Feel free to:
- Experiment with different configurations
- Add new validation rules
- Implement additional constraints
- Share improvements

## 📝 License

This demo follows the licenses of its dependencies:
- Phi-3: MIT License
- Transformers, PEFT, TRL: Apache 2.0
- CADmium: Check dataset license on Hugging Face

## 🙏 Acknowledgments

- Microsoft Research for Phi-3
- Hugging Face for PEFT and TRL libraries
- CADmium dataset creators
- Open source community

---

**Happy Fine-tuning! 🚀**

For questions or issues, refer to the inline comments in `LoRADemo.ipynb` or check the troubleshooting section above.
