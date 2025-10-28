# Zoo Gym - Unified AI Model Training Platform

**Category**: Hanzo Ecosystem
**Difficulty**: Intermediate
**Prerequisites**: Python 3.10+, PyTorch, basic ML knowledge
**Related Skills**: zenlm.md, python-sdk.md, hanzo-dev.md

## Overview

**Zoo Gym** is the unified training infrastructure for all ZenLM AI models, built on LLaMA Factory. Developed by Zoo Labs Foundation (501(c)(3) charitable organization dedicated to AI research for wildlife conservation and preservation), it provides comprehensive support for fine-tuning, reinforcement learning, and quantization across the entire ZenLM model family—from zen-nano (0.6B) to zen-musician (7B).

**Why Zoo Gym?**
- **Unified Platform**: Train all ZenLM models with consistent tooling
- **Production-Grade**: Battle-tested methods including GRPO (Group Relative Policy Optimization)
- **Training-Free GRPO Innovation**: Distinct implementation based on Tencent's youtu-agent research - no value network needed
- **Performance Optimized**: 2-5x faster with Unsloth, FlashAttention-2, Liger Kernel
- **Hardware Efficient**: Train 4B models on 8GB GPUs with QLoRA
- **Export Ready**: GGUF, MLX, AWQ, GPTQ quantization for deployment
- **Wildlife Conservation**: Part of Zoo Labs Foundation's mission to aid conservation efforts

## Quick Start

### Installation

```bash
# Clone Zoo Gym repository
git clone https://github.com/zooai/gym.git
cd gym

# Create environment
conda create -n zoo-gym python=3.10
conda activate zoo-gym
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
# Launch Zoo Gym web interface
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
**Innovation**: Zoo Gym's distinct training-free GRPO implementation - no value network required
**Memory**: 40-60% less than PPO (eliminates value network overhead)
**Speed**: 2x faster than PPO
**Quality**: Superior to DPO for instruction following

**Implementation**: Zoo Gym uses a distinct training-free GRPO variant based on [Tencent's youtu-agent research](https://github.com/TencentCloudADP/youtu-agent/tree/training_free_GRPO/training_free_grpo). This eliminates the value network entirely, unlike standard GRPO implementations (e.g., DeepSeek), making it truly training-free and more memory-efficient for RLHF workflows.

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

**Note on Training-Free GRPO**: Training-free GRPO is the **default approach** used in Zoo Gym for local LLMs and production deployments. It provides the best balance of memory efficiency and quality without requiring value networks. For specialized use cases like per-user personalization, see BitDelta/DeltaSoup below.

## Per-User Personalization (BitDelta & DeltaSoup)

### BitDelta (ZIP-7) - Personalized Models Without LoRA

**Purpose**: Create millions of personalized model variants for individual users **without** using LoRA
**Innovation**: 1-bit quantization of fine-tune deltas - **distinct from training-free GRPO**
**Memory**: 10× reduction for personalized models
**Safety**: 60% reduction in jailbreak risks
**Use Case**: Per-user personalization of frontier LLMs (GPT-4, Claude, Llama, Qwen)

**How It Works**:
- Compresses fine-tune deltas to binary signs + scales
- Each user gets a unique personalized variant
- Base model stays in memory, deltas are 1-bit
- Serve thousands of users from single GPU

```bash
# Train personalized variant for user
llamafactory-cli train \
    --stage sft \
    --model_name_or_path Qwen/Qwen3-4B \
    --dataset user_123_preferences \
    --template qwen3 \
    --finetuning_type bitdelta \
    --output_dir ./variants/user_123 \
    --quantization_method bitdelta \
    --bits 1 \
    --group_size 128 \
    --safety_threshold 0.6
```

**Python SDK Integration**:

```python
from zoo.gym import PersonalizedTrainer, BitDeltaConfig

# Configure BitDelta for per-user personalization
config = BitDeltaConfig(
    bits=1,                    # 1-bit quantization
    group_size=128,            # Group size for quantization
    safety_threshold=0.6,      # 60% jailbreak reduction
    enable_deltasoup=True      # Enable community aggregation
)

# Create personalized trainer
trainer = PersonalizedTrainer(
    base_model="Qwen/Qwen3-4B",
    config=config
)

# Train personalized variants
trainer.create_variant(
    user_id="user_123",
    preferences=user_preferences,
    dataset=user_dataset
)

# Serve variant (10× memory efficient)
response = trainer.serve_variant(
    user_id="user_123",
    prompt="Hello, what do you remember about me?"
)
```

### DeltaSoup - Community-Driven Improvement

**Purpose**: Aggregate personalized improvements from multiple users
**Innovation**: Byzantine-robust community learning
**Safety**: Differential privacy + outlier rejection
**Rewards**: Contributor rewards based on quality

**How It Works**:
- Users contribute their personalized deltas
- Byzantine-robust aggregation filters malicious contributions
- Differential privacy protects individual contributions
- Contributors earn rewards based on quality

```python
from zoo.gym import DeltaSoup, AggregationMethod, DeltaSoupConfig

# Configure DeltaSoup
config = DeltaSoupConfig(
    method=AggregationMethod.BYZANTINE_ROBUST,
    differential_privacy=True,
    enable_rewards=True,
    min_contributors=3,
    quality_threshold=0.8
)

# Create soup
soup = DeltaSoup(config)

# Users contribute their improvements
soup.contribute(user_id="alice", model=model_alice)
soup.contribute(user_id="bob", model=model_bob)
soup.contribute(user_id="charlie", model=model_charlie)

# Aggregate improvements
aggregated_model = soup.aggregate()

# Distribute rewards
rewards = soup.calculate_rewards()
# {'alice': 42.5, 'bob': 38.2, 'charlie': 19.3}
```

### BitDelta vs LoRA vs Training-Free GRPO

| Method | Purpose | Memory | Speed | Use Case |
|--------|---------|---------|-------|----------|
| **Training-Free GRPO** | RLHF | -40% vs PPO | 2x vs PPO | **Default for local LLMs & production** |
| **BitDelta** | Per-user personalization | 10× reduction | Fast | Millions of user variants |
| **DeltaSoup** | Community learning | Efficient | Async | Aggregate user improvements |
| **LoRA** | General fine-tuning | -70% vs full | 1.5x vs full | Single-model adaptation |
| **QLoRA** | Memory-constrained | -90% vs full | 1.2x vs full | Train 4B on 8GB GPU |

**Key Distinctions**:
- **Training-Free GRPO**: Used everywhere by default in Zoo Gym for local and production LLMs (no value network)
- **BitDelta**: For per-user personalization without LoRA (works directly on fine-tune deltas)
- **DeltaSoup**: For community-driven improvements (aggregates user contributions)
- **LoRA/QLoRA**: For general-purpose fine-tuning (adapter layers)

## Cost Savings: Training-Free GRPO vs Traditional Fine-Tuning

**Dramatic cost reduction** using Training-Free GRPO compared to traditional LoRA or standard RL methods:

### Training Cost Comparison

| Method | Model | Training Samples | Training Cost | Performance (AIME24) |
|--------|-------|-----------------|---------------|---------------------|
| **Training-Free GRPO** | DeepSeek-V3.1-Terminus (671B) | 100 | **~$18** | **82.7%** |
| ReTool (Standard RL) | Qwen2.5-32B-Instruct | 1000s | **~$10,000** | 67.0% |
| LoRA Fine-Tuning | Qwen2.5-32B-Instruct | 1000s | **~$20,000** | ~60-70% |

**Key Findings**:
- **500× cost reduction**: $18 vs $10,000+ for superior performance
- **100× fewer samples**: 100 vs 1000s of training examples needed
- **Better results**: 82.7% vs 67.0% on AIME24 benchmark
- **No parameter updates**: Zero gradient computation, instant deployment
- **Larger model advantage**: 671B frozen model outperforms fine-tuned 32B

### Infrastructure Cost Breakdown

**Traditional LoRA/RL Training** (Example: ReTool on Qwen2.5-32B):
```
Training Cost:
- GPU hours: 20,000 × $0.5/hour = $10,000
- Data collection: $2,000-5,000
- Total: $12,000-15,000

Deployment Cost (Dedicated GPU):
- 4× GPUs at $0.5/GPU-hour = $2/hour = $1,440/month
- Inference: ~$0.005 per problem (400 problems/hour)
- Fixed infrastructure: Always running, even with low usage

Result: $10K+ training + $1.4K/month deployment
```

**Training-Free GRPO** (Example: DeepSeek-V3.1-Terminus):
```
Training Cost:
- API calls: 38M input tokens + 6.6M output tokens
- With cache hit pricing: ~$18 for 100 samples
- 3 training steps over 6 hours
- Total: $18

Deployment Cost (Pay-as-you-go API):
- ~$0.02 per problem (60K input + 8K output tokens)
- No fixed infrastructure
- Only pay for actual usage
- Automatic scaling, no GPU management

Result: $18 training + pay-per-use inference
```

### When Training-Free GRPO Saves More

**Ideal scenarios for maximum cost savings**:

1. **Low-frequency applications**: Don't need dedicated GPU cluster
2. **Specialized domains**: Need adaptation but not full fine-tuning
3. **Limited data**: Only dozens of examples available
4. **Rapid iteration**: Need quick experiments without multi-day training
5. **Multiple domains**: Switch between domains by plugging in different experiences
6. **Budget constraints**: Cannot afford $10K+ training costs

### Real-World Cost Example

**Scenario**: Medical diagnosis assistant (low-traffic, specialized domain)

**Traditional LoRA Fine-Tuning**:
- Training: $15,000 (data collection + GPU time)
- Deployment: $1,440/month (dedicated GPU)
- First year: $15,000 + $17,280 = **$32,280**
- Rigid specialization: Cannot adapt to new medical domains without retraining

**Training-Free GRPO**:
- Training: $50 (200 samples across 3 medical subdomains)
- Deployment: $200/month (100 diagnoses/day @ $0.02 each, 100 days/month)
- First year: $50 + $2,400 = **$2,450**
- Flexible: Switch medical domains by plugging in different experiences
- **Savings**: $29,830 (92% cost reduction)

### SDK Support for Training-Free GRPO

**Full Python SDK support**:
```python
from zoo.gym import TrainingFreeGRPO, GRPOConfig

# Configure Training-Free GRPO
config = GRPOConfig(
    group_size=5,              # 5 rollouts per query
    max_epochs=3,              # 3 training steps
    temperature_train=0.7,     # Training temperature
    temperature_eval=0.3       # Evaluation temperature
)

# Create trainer
trainer = TrainingFreeGRPO(
    base_model="deepseek/v3.1-terminus",
    config=config
)

# Train with minimal samples
experiences = trainer.train(
    dataset=math_problems[:100],  # Only 100 samples!
    reward_model=reward_fn
)

# Deploy instantly - no parameter updates needed
response = trainer.infer(
    query="Solve this geometry problem...",
    experiences=experiences  # Plug in learned experiences
)
```

**Full Rust SDK support**:
```rust
use zoo_gym::{TrainingFreeGRPO, GRPOConfig};

let config = GRPOConfig {
    group_size: 5,
    max_epochs: 3,
    temperature_train: 0.7,
    temperature_eval: 0.3,
};

let trainer = TrainingFreeGRPO::new("deepseek/v3.1-terminus", config);
let experiences = trainer.train(&dataset[..100], reward_fn)?;
let response = trainer.infer(query, &experiences)?;
```

### Native Hanzo Ecosystem Support

**Hanzo Node**: Serve Training-Free GRPO models locally
- Load frozen base model (e.g., DeepSeek, Qwen)
- Inject learned experiences as context
- Zero-latency experience switching
- Local inference for privacy

**Hanzo Engine**: Optimized inference with experience injection
- Native Rust implementation
- 1-bit BitDelta variants supported
- Memory-efficient experience caching
- Multi-experience parallel serving

**Hanzo Dev**: CLI training workflows
```bash
# Train with Training-Free GRPO
dev "train training-free GRPO on math dataset with 100 samples"

# Export experiences
dev "export learned experiences to JSON"

# Deploy to local node
dev "deploy experiences to hanzo node with deepseek base model"
```

**Hanzo Desktop**: Local training and mining
- Train Training-Free GRPO on MacBook
- Mine $AI tokens while serving personalized models
- ZenLM models (zen-eco, zen-agent) with learned experiences
- Automatic experience caching and reuse

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

## Comparison: Manual Training vs Zoo Gym

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

### Zoo Gym (1 command)

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

## Integration with Hanzo/ZooLM Ecosystem

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
# Launch Zoo Gym GUI for visual configuration
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

- **zenlm.md**: Learn about ZenLM model family and capabilities
- **python-sdk.md**: Deploy trained models with Hanzo SDK
- **hanzo-dev.md**: Use terminal agent to automate training workflows
- **hanzo-desktop.md**: Run trained models locally with mining capabilities

## Additional Resources

- **GitHub**: https://github.com/zooai/gym
- **Zoo Labs Foundation**: https://zoo.ngo (501(c)(3) charitable organization)
- **LLaMA Factory**: https://github.com/hiyouga/LLaMA-Factory
- **HuggingFace Models**: https://huggingface.co/zoolm

## Zoo Labs Foundation

Zoo Gym is developed and maintained by **Zoo Labs Foundation** (501(c)(3) tax-exempt charitable organization) dedicated to AI research for wildlife conservation, preservation, and environmental protection.

**Tax ID**: 88-3538992
**Mission**: Leverage AI technology to aid in wildlife conservation efforts globally

By using Zoo Gym, you're supporting open-source AI research that directly contributes to environmental conservation.

---

**Zoo Gym** - Train any ZenLM model with production-grade efficiency.
Developed by **Zoo Labs Foundation** for conservation through AI.