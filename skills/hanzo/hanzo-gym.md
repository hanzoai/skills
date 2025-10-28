# Hanzo Gym - Unified AI Model Training Platform

**Category**: Hanzo Ecosystem
**Difficulty**: Intermediate
**Prerequisites**: Python 3.10+, PyTorch, basic ML knowledge
**Related Skills**: zenlm.md, python-sdk.md, hanzo-dev.md

## Overview

Hanzo Gym (Zen Gym) is the unified training infrastructure for all Zen AI models, built on LLaMA Factory. It provides comprehensive support for fine-tuning, reinforcement learning, and quantization across the entire Zen model family—from zen-nano (0.6B) to zen-musician (7B).

**Why Hanzo Gym?**
- **Unified Platform**: Train all Zen models with consistent tooling
- **Production-Grade**: Battle-tested methods (LoRA, QLoRA, GRPO, GSPO, DPO, PPO)
- **Performance Optimized**: 2-5x faster with Unsloth, FlashAttention-2, Liger Kernel
- **Hardware Efficient**: Train 4B models on 8GB GPUs with QLoRA
- **Export Ready**: GGUF, MLX, AWQ, GPTQ quantization for deployment

## Quick Start

### Installation

```bash
cd /Users/z/work/zen/gym
conda create -n zen-gym python=3.10
conda activate zen-gym
pip install -r requirements.txt

# FlashAttention-2 (recommended - 2x faster)
pip install flash-attn --no-build-isolation

# Unsloth acceleration (2-5x faster training)
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

### Train Your First Model (zen-nano with LoRA)

```bash
llamafactory-cli train \
    --stage sft \
    --do_train \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --dataset your_dataset \
    --template qwen3 \
    --finetuning_type lora \
    --lora_target all \
    --output_dir ./zen-nano-lora \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 3 \
    --lr_scheduler_type cosine \
    --learning_rate 5e-5 \
    --flash_attn fa2 \
    --use_unsloth true \
    --save_steps 100 \
    --logging_steps 10
```

### GUI Training Interface

```bash
# Launch Zen Gym web interface
llamafactory-cli webui
```

Access at http://localhost:7860 for visual training configuration.

## Training Methods

### 1. LoRA (Low-Rank Adaptation)

**Best for**: Most training scenarios
**Memory**: ~30% of full fine-tuning
**Speed**: 1.5-2x faster
**Quality**: 95-98% of full fine-tuning

```bash
llamafactory-cli train \
    --stage sft \
    --do_train \
    --model_name_or_path Qwen/Qwen3-4B \
    --dataset your_dataset \
    --template qwen3 \
    --finetuning_type lora \
    --lora_rank 64 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --lora_target all \
    --output_dir ./model-lora \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5
```

### 2. QLoRA (Quantized LoRA)

**Best for**: Limited GPU memory (train 4B on 8GB GPU)
**Memory**: ~10% of full fine-tuning
**Speed**: 1.2-1.5x faster than full
**Quality**: 90-95% of full fine-tuning

```bash
llamafactory-cli train \
    --stage sft \
    --do_train \
    --model_name_or_path Qwen/Qwen3-4B \
    --dataset your_dataset \
    --template qwen3 \
    --finetuning_type lora \
    --quantization_bit 4 \
    --lora_rank 64 \
    --lora_alpha 32 \
    --output_dir ./model-qlora \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 2e-4
```

### 3. GRPO (Group Relative Policy Optimization)

**Best for**: Reinforcement learning from human feedback
**Memory**: 40-60% less than PPO (no value network)
**Speed**: 2x faster than PPO
**Quality**: Superior to DPO for instruction following

```bash
llamafactory-cli train \
    --stage grpo \
    --do_train \
    --model_name_or_path Qwen/Qwen3-4B \
    --dataset your_preference_dataset \
    --template qwen3 \
    --finetuning_type lora \
    --output_dir ./model-grpo \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 1 \
    --learning_rate 1e-5
```

### 4. GSPO (Group Sampled Policy Optimization)

**Best for**: Mixture-of-Experts models (Qwen3-MoE)
**Memory**: Similar to GRPO
**Speed**: Optimized for MoE stability

```bash
llamafactory-cli train \
    --stage gspo \
    --do_train \
    --model_name_or_path Qwen/Qwen3-4B-MoE \
    --dataset your_preference_dataset \
    --template qwen3 \
    --finetuning_type lora \
    --output_dir ./model-gspo \
    --learning_rate 1e-5
```

## Zen Model Training Configs

### zen-nano (0.6B) - Ultra-lightweight

```yaml
model_name_or_path: Qwen/Qwen3-0.6B
template: qwen3
finetuning_type: lora
lora_rank: 64
lora_alpha: 32
per_device_train_batch_size: 8
gradient_accumulation_steps: 2
learning_rate: 5e-5
flash_attn: fa2
use_unsloth: true
max_length: 2048
num_train_epochs: 3
```

**Hardware Requirements**: 4GB GPU (GTX 1660 Ti+)

### zen-eco (4B) - Efficient models (instruct/thinking/agent)

```yaml
model_name_or_path: Qwen/Qwen3-4B
template: qwen3
finetuning_type: lora
lora_rank: 128
lora_alpha: 64
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
learning_rate: 2e-5
flash_attn: fa2
use_unsloth: true
max_length: 4096
num_train_epochs: 3
```

**Hardware Requirements**: 16GB GPU (RTX 4060 Ti+) or 8GB with QLoRA

### zen-agent (4B) - Tool calling & function execution

```yaml
model_name_or_path: Qwen/Qwen3-4B
dataset: Salesforce/xlam-function-calling-60k
template: qwen3
finetuning_type: lora
lora_rank: 128
lora_alpha: 64
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
learning_rate: 1e-5
max_length: 8192
```

**Hardware Requirements**: 16GB GPU (RTX 4060 Ti+)

### zen-musician (7B) - Music generation with lyrics

```yaml
model_name_or_path: m-a-p/YuE-s1-7B-anneal-en-cot
template: yue
finetuning_type: lora
lora_rank: 64
lora_alpha: 32
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
learning_rate: 2e-4
flash_attn: fa2
max_length: 4096
```

**Hardware Requirements**: 24GB GPU (RTX 3090+) or 12GB with QLoRA

## Quantization & Export

### Export to GGUF (llama.cpp)

```bash
# Q4_K_M (recommended for most use cases)
llamafactory-cli export \
    --model_name_or_path ./zen-eco-lora \
    --adapter_name_or_path ./zen-eco-lora \
    --template qwen3 \
    --export_dir ./zen-eco-gguf \
    --export_size 4 \
    --export_quantization_bit 4 \
    --export_legacy_format false

# Q8_0 (higher quality)
llamafactory-cli export \
    --model_name_or_path ./model \
    --export_dir ./gguf \
    --export_quantization_bit 8

# Q2_K (maximum compression for mobile)
llamafactory-cli export \
    --model_name_or_path ./model \
    --export_dir ./gguf \
    --export_quantization_bit 2
```

### MLX Conversion (Apple Silicon)

```bash
# Convert to MLX format for Mac
python -m mlx_lm.convert \
    --hf-path ./zen-eco-lora \
    --mlx-path ./zen-eco-mlx \
    --quantize
```

## Performance Optimizations

### FlashAttention-2

**Benefit**: 2x faster attention, 22GB → 20GB memory
**Installation**: `pip install flash-attn --no-build-isolation`

```bash
--flash_attn fa2
```

### Unsloth

**Benefit**: 2-5x faster training, 20GB → 18GB memory
**Installation**: `pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"`

```bash
--use_unsloth true
```

### Liger Kernel

**Benefit**: 3x faster, LinkedIn's optimized kernels
**Installation**: Included in requirements.txt

```bash
--enable_liger_kernel true
```

### Gradient Checkpointing

**Benefit**: 2.8x speed, 18GB → 14GB memory (trades compute for memory)

```bash
--gradient_checkpointing true
```

### Combined Optimization

```bash
llamafactory-cli train \
    --model_name_or_path Qwen/Qwen3-4B \
    --flash_attn fa2 \
    --use_unsloth true \
    --enable_liger_kernel true \
    --gradient_checkpointing true \
    --quantization_bit 4
```

**Result**: Train zen-eco-4B on 8GB GPU at 3x baseline speed!

## Hardware Requirements

| Model Size | Training Method | GPU Memory | Recommended GPU | Speed |
|------------|----------------|------------|-----------------|-------|
| 0.6B | Full | 8GB | RTX 3060 | 1.0x |
| 0.6B | LoRA | 4GB | GTX 1660 Ti | 1.5x |
| 4B | Full | 32GB | RTX 3090 | 1.0x |
| 4B | LoRA | 16GB | RTX 4060 Ti | 1.5x |
| 4B | QLoRA 4-bit | 8GB | RTX 3060 | 1.2x |
| 4B | LoRA + Unsloth | 14GB | RTX 3060 Ti | 2.5x |
| 7B | Full | 48GB | A6000 | 1.0x |
| 7B | LoRA | 24GB | RTX 3090 | 1.5x |
| 7B | QLoRA 4-bit | 12GB | RTX 3060 Ti | 1.2x |

## Experiment Tracking

### Weights & Biases

```bash
export WANDB_PROJECT=zen-models
export WANDB_API_KEY=your_key

llamafactory-cli train \
    --report_to wandb \
    --config your_config.yaml
```

### TensorBoard (Built-in)

```bash
# TensorBoard logs automatically created
tensorboard --logdir ./output
```

### MLflow

```bash
llamafactory-cli train \
    --report_to mlflow \
    --config your_config.yaml
```

## Deployment

### OpenAI-style API (vLLM)

```bash
llamafactory-cli api \
    --model_name_or_path ./zen-eco-lora \
    --template qwen3 \
    --infer_backend vllm \
    --port 8000
```

### Gradio Chat Interface

```bash
llamafactory-cli chat \
    --model_name_or_path ./zen-eco-lora \
    --template qwen3
```

Access at http://localhost:7860

## Troubleshooting

### Out of Memory Errors

```bash
# 1. Reduce batch size
--per_device_train_batch_size 1

# 2. Increase gradient accumulation
--gradient_accumulation_steps 32

# 3. Enable gradient checkpointing
--gradient_checkpointing true

# 4. Use QLoRA 4-bit
--quantization_bit 4
```

### Slow Training

```bash
# 1. Enable FlashAttention-2
--flash_attn fa2

# 2. Enable Unsloth
--use_unsloth true

# 3. Enable Liger Kernel
--enable_liger_kernel true

# 4. Check GPU utilization
nvidia-smi -l 1
```

### Model Divergence (Loss exploding)

```bash
# 1. Lower learning rate
--learning_rate 1e-5

# 2. Add warmup
--warmup_ratio 0.1

# 3. Use cosine scheduler
--lr_scheduler_type cosine

# 4. Reduce gradient accumulation
--gradient_accumulation_steps 4
```

## Comparison: Manual Training vs Hanzo Gym

### Manual PyTorch Training (50-100+ lines)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")

# Configure LoRA
lora_config = LoraConfig(
    r=64,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# Load and preprocess dataset
dataset = load_dataset("your_dataset")
def preprocess_function(examples):
    # Complex tokenization logic...
    pass
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Configure trainer with custom settings
training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    num_train_epochs=3,
    lr_scheduler_type="cosine",
    # Many more parameters...
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer
)

# Train and export
trainer.train()
model.save_pretrained("./output")

# Manual GGUF export (complex multi-step process)
# ... 20+ more lines ...
```

### Hanzo Gym (1 command)

```bash
llamafactory-cli train \
    --stage sft \
    --do_train \
    --model_name_or_path Qwen/Qwen3-4B \
    --dataset your_dataset \
    --template qwen3 \
    --finetuning_type lora \
    --lora_target all \
    --output_dir ./zen-eco-lora \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 3 \
    --learning_rate 5e-5 \
    --flash_attn fa2 \
    --use_unsloth true
```

**Result**: 50-100 lines → 1 command, **10-50x faster development**

## Integration with Hanzo Ecosystem

### Train with Hanzo Python SDK

```python
from hanzo import Hanzo

# Initialize Hanzo with local training mode
hanzo = Hanzo(
    inference_mode='local',
    node_url='http://localhost:8080'
)

# Deploy trained model to Hanzo Node
hanzo.deploy_model(
    model_path='./zen-eco-lora',
    model_name='zen-eco-custom',
    quantization='4bit'
)

# Use trained model immediately
response = hanzo.chat.completions.create(
    model='zen-eco-custom',
    messages=[{'role': 'user', 'content': 'Test my fine-tuned model'}]
)
```

### Use with Hanzo Dev (Terminal Agent)

```bash
# Train model
dev "train zen-eco with my dataset using LoRA"

# Export to GGUF
dev "export the trained model to GGUF Q4_K_M"

# Deploy to Hanzo Node
dev "deploy the GGUF model to my local Hanzo Node"
```

## Best Practices

### 1. Start Small, Scale Up

```bash
# 1. Test with zen-nano first (fast iteration)
llamafactory-cli train --model_name_or_path Qwen/Qwen3-0.6B ...

# 2. Move to zen-eco when ready
llamafactory-cli train --model_name_or_path Qwen/Qwen3-4B ...

# 3. Scale to zen-musician for production
llamafactory-cli train --model_name_or_path m-a-p/YuE-s1-7B ...
```

### 2. Use GUI for Experimentation

```bash
# Launch GUI for visual configuration
llamafactory-cli webui

# Export config once satisfied
# config.yaml created automatically
```

### 3. Track Everything

```bash
# Use WandB for professional tracking
export WANDB_PROJECT=my-zen-models
llamafactory-cli train --report_to wandb ...
```

### 4. Optimize Progressively

```bash
# Baseline (no optimization)
llamafactory-cli train --model_name_or_path Qwen/Qwen3-4B ...

# Add FlashAttention-2 (+50% speed)
--flash_attn fa2

# Add Unsloth (+100-150% speed)
--use_unsloth true

# Add gradient checkpointing (if memory constrained)
--gradient_checkpointing true
```

## Related Skills

- **zenlm.md**: Learn about Zen model family and capabilities
- **python-sdk.md**: Deploy trained models with Hanzo SDK
- **hanzo-dev.md**: Use terminal agent to automate training workflows
- **hanzo-desktop.md**: Run trained models locally with mining capabilities

## Additional Resources

- **Official Docs**: https://gym.readthedocs.io/
- **Training Configs**: `/Users/z/work/zen/gym/configs/`
- **Qwen3 Guide**: `/Users/z/work/zen/gym/configs/qwen3_training_guide.md`
- **LLaMA Factory**: https://github.com/hiyouga/LLaMA-Factory

---

**Hanzo Gym** - Train any Zen model with production-grade efficiency.
Part of the **Hanzo AI** ecosystem.