# Training-Free GRPO - RLHF Without Value Networks

**Category**: Zoo Gym Training Methods
**Skill Level**: Advanced
**Prerequisites**: Understanding of RLHF, PPO, policy optimization, reward modeling
**Related Skills**: bitdelta.md, deltasoup.md, ../hanzo/hanzo-gym.md
**Zoo Gym Default**: This is the default RLHF approach used in Zoo Gym

## Overview

Training-Free GRPO (Group Relative Policy Optimization) is Zoo Gym's **default RLHF method** that eliminates value networks entirely, achieving 40-60% memory reduction and 2× speedup compared to PPO. It's based on DeepSeek's GRPO innovation and **distinct from BitDelta/DeltaSoup** which focus on per-user personalization and community learning.

**Core Innovation**: Group sampling + relative advantage estimation within groups = **no value network needed** = massive memory savings + faster training.

## Why Training-Free GRPO?

### The Problem: Traditional RLHF (PPO) is Expensive

**PPO Requirements**:
- **Policy model**: The LLM being trained (~10GB for 4B model)
- **Value network**: Separate network to estimate value function (~10GB)
- **Reference model**: Frozen copy for KL penalty (~10GB)
- **Total**: 30GB+ GPU memory for 4B model

**PPO Training**:
- Compute advantage: `A = Q(s,a) - V(s)`
- Requires value network V(s) trained simultaneously
- Value network training adds 40-60% overhead
- Memory bottleneck limits batch sizes

### The Training-Free GRPO Solution

**GRPO Requirements**:
- **Policy model**: The LLM being trained (~10GB)
- **Reference model**: Frozen copy for KL penalty (~10GB)
- **Value network**: **NONE** ❌
- **Total**: 20GB GPU memory (40% less than PPO)

**GRPO Training**:
- Group sampling: Generate K responses per query
- Relative advantage: `A = R_i - mean(R_group)`
- No value network needed!
- 40-60% memory savings, 2× speedup

## How Training-Free GRPO Works

### 1. Group Sampling

Generate multiple responses (K=5-8) for each query:

```python
# Traditional PPO: 1 response per query
query = "Solve: 2x + 3 = 11"
response = model.generate(query)
reward = reward_model(query, response)

# Training-Free GRPO: K responses per query
K = 5
responses = [model.generate(query) for _ in range(K)]
rewards = [reward_model(query, r) for r in responses]

# Group:
# responses[0]: "x = 4" (correct) → reward = 1.0
# responses[1]: "x = 3" (wrong) → reward = 0.0
# responses[2]: "x = 4" (correct) → reward = 1.0
# responses[3]: "x = 5" (wrong) → reward = 0.0
# responses[4]: "x = 4" (correct) → reward = 1.0
```

### 2. Relative Advantage Estimation

Compute advantages relative to group mean:

```python
# PPO: Absolute advantage (requires value network)
# A = Q(s,a) - V(s)
# V(s) = value_network(state) ← REQUIRES VALUE NETWORK

# GRPO: Relative advantage (no value network)
mean_reward = sum(rewards) / K  # 0.6
advantages = [r - mean_reward for r in rewards]

# Advantages:
# [1.0 - 0.6 = 0.4]  # Good response
# [0.0 - 0.6 = -0.6] # Bad response
# [1.0 - 0.6 = 0.4]  # Good response
# [0.0 - 0.6 = -0.6] # Bad response
# [1.0 - 0.6 = 0.4]  # Good response
```

### 3. Policy Optimization

Update policy using relative advantages:

```python
# Compute policy loss (same as PPO, but advantages are relative)
for response, advantage in zip(responses, advantages):
    # Log probability under current policy
    log_prob = model.log_prob(query, response)

    # Log probability under reference policy (for KL penalty)
    ref_log_prob = ref_model.log_prob(query, response)

    # Ratio
    ratio = torch.exp(log_prob - ref_log_prob)

    # Clipped surrogate loss
    loss = -torch.min(
        ratio * advantage,
        torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * advantage
    )

    # Backprop and update
    loss.backward()
    optimizer.step()
```

### 4. Why This Works

**Key insight**: Relative advantages are sufficient for policy optimization

```python
# PPO (absolute advantage):
# A = Q(s,a) - V(s)
# "How much better is this action than average?"

# GRPO (relative advantage):
# A = R(s,a) - mean(R_group)
# "How much better is this response than other responses in the group?"

# Both capture the same signal!
# GRPO doesn't need V(s) because it's comparing within groups
```

## Memory Comparison

| Component | PPO | GRPO | Savings |
|-----------|-----|------|---------|
| Policy Model | 10GB | 10GB | 0% |
| Value Network | 10GB | **0GB** | **100%** |
| Reference Model | 10GB | 10GB | 0% |
| **Total** | **30GB** | **20GB** | **40%** |

## Training Speed Comparison (Qwen3-4B on 4× A100)

| Method | Training Time | Tokens/sec | Memory |
|--------|---------------|------------|--------|
| Full Fine-Tune | 12 hours | 500 | 40GB |
| PPO | 8 hours | 800 | 30GB |
| **GRPO** | **4 hours** | **1600** | **20GB** |
| LoRA | 2 hours | 2000 | 15GB |

**Note**: LoRA is faster but doesn't do RLHF (only supervised fine-tuning).

## Cost Savings: Training-Free GRPO vs Traditional Methods

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

## Installation

```bash
# Zoo Gym includes Training-Free GRPO by default
git clone https://github.com/zooai/gym.git
cd gym
pip install -e .

# Or via pip
pip install zoo-gym
```

## Usage Examples

### Basic Training-Free GRPO

```python
from zoo.gym import GRPOTrainer, GRPOConfig

# Configure GRPO
config = GRPOConfig(
    group_size=5,              # 5 rollouts per query
    max_epochs=3,              # 3 training epochs
    beta=0.1,                  # KL penalty coefficient
    clip_range=0.2,            # PPO-style clipping
    normalize_advantages=True, # Normalize advantages
    temperature_train=0.7,     # Training temperature
    temperature_eval=0.3       # Evaluation temperature
)

# Create trainer (no value network!)
trainer = GRPOTrainer(
    model=model,
    ref_model=ref_model,
    tokenizer=tokenizer,
    config=config,
    reward_model=reward_fn
)

# Train with minimal samples
trainer.train(
    train_dataset=train_dataset,  # Only 100-1000 samples needed
    eval_dataset=eval_dataset
)

# Save trained model
trainer.save_model("./trained_model")
```

### Training-Free GRPO with Experience Injection

```python
from zoo.gym import TrainingFreeGRPO, GRPOConfig

# Configure Training-Free GRPO
config = GRPOConfig(
    group_size=5,
    max_epochs=3,
    temperature_train=0.7,
    temperature_eval=0.3
)

# Create trainer (frozen base model)
trainer = TrainingFreeGRPO(
    base_model="deepseek/v3.1-terminus",  # Frozen 671B model
    config=config
)

# Train with minimal samples (100 is enough!)
experiences = trainer.train(
    dataset=math_problems[:100],
    reward_model=reward_fn
)

# Deploy instantly - no parameter updates needed
response = trainer.infer(
    query="Solve: 2x + 3 = 11",
    experiences=experiences  # Plug in learned experiences
)
print(response)  # "x = 4"
```

### CLI Training with GRPO

```bash
# Train with Training-Free GRPO (default in Zoo Gym)
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
    --learning_rate 1e-5 \
    --grpo_group_size 8 \
    --grpo_beta 0.1 \
    --grpo_clip_range 0.2
```

### Custom Reward Function

```python
def custom_reward_fn(query: str, response: str) -> float:
    """Custom reward function for math problems"""
    # Parse expected answer
    expected = parse_answer(query)

    # Parse model answer
    predicted = parse_answer(response)

    # Binary reward
    if predicted == expected:
        return 1.0
    else:
        return 0.0

# Use custom reward
trainer = GRPOTrainer(
    model=model,
    ref_model=ref_model,
    config=config,
    reward_model=custom_reward_fn  # Custom reward
)
```

## Advanced Features

### 1. Group Size Tuning

```python
# Larger groups = more stable but slower
config = GRPOConfig(group_size=10)  # 10 responses per query

# Smaller groups = faster but less stable
config = GRPOConfig(group_size=3)  # 3 responses per query

# Recommended: 5-8 for most tasks
config = GRPOConfig(group_size=5)  # Default
```

### 2. Temperature Scheduling

```python
# Use higher temperature during training (exploration)
# Use lower temperature during evaluation (exploitation)
config = GRPOConfig(
    temperature_train=0.9,  # High exploration
    temperature_eval=0.1    # Low exploitation
)
```

### 3. KL Penalty Tuning

```python
# Beta controls KL penalty (how much to stay close to reference model)
config = GRPOConfig(
    beta=0.01  # Weak penalty, more exploration
)

config = GRPOConfig(
    beta=0.5   # Strong penalty, stay close to reference
)
```

### 4. Integration with BitDelta

```python
from zoo.gym import GRPOTrainer, BitDeltaConfig

# Train with GRPO + BitDelta compression
bitdelta_config = BitDeltaConfig(
    bits=1,
    group_size=128,
    safety_threshold=0.6
)

grpo_config = GRPOConfig(group_size=5, max_epochs=3)

# Train (GRPO) + Compress (BitDelta) + Serve
trainer = GRPOTrainer(model, ref_model, grpo_config)
trainer.train(train_dataset)
trainer.compress_deltas(bitdelta_config)  # 10× compression
trainer.save_model("./compressed_model")
```

## Performance Benchmarks

### AIME24 Math Benchmark

| Method | Model | Training Samples | Accuracy | Cost |
|--------|-------|-----------------|----------|------|
| **Training-Free GRPO** | DeepSeek-V3.1 (671B) | 100 | **82.7%** | **$18** |
| ReTool (Standard RL) | Qwen2.5 (32B) | 1000s | 67.0% | $10,000 |
| LoRA Fine-Tuning | Qwen2.5 (32B) | 1000s | ~65% | $20,000 |
| Base Model (zero-shot) | Qwen2.5 (32B) | 0 | 45% | $0 |

### Memory Efficiency (Training Qwen3-4B)

| Method | GPU Memory | Max Batch Size |
|--------|------------|----------------|
| Full Fine-Tune | 40GB | 1 |
| PPO (with value network) | 30GB | 2 |
| **GRPO (no value network)** | **20GB** | **4** |
| LoRA | 15GB | 8 |

### Training Speed (4× A100 GPUs)

| Method | Tokens/sec | Training Time | Cost |
|--------|------------|---------------|------|
| Full Fine-Tune | 500 | 12 hours | $240 |
| PPO | 800 | 8 hours | $160 |
| **GRPO** | **1600** | **4 hours** | **$80** |
| LoRA | 2000 | 2 hours | $40 |

## GRPO vs PPO vs LoRA vs BitDelta

| Method | Purpose | Memory | Speed | Use Case |
|--------|---------|--------|-------|----------|
| **Training-Free GRPO** | RLHF without value network | -40% vs PPO | 2× vs PPO | **Default for RLHF in Zoo Gym** |
| **PPO** | Traditional RLHF | Baseline | Baseline | Research, legacy systems |
| **LoRA** | Supervised fine-tuning | -60% vs full | 4× vs full | Instruction tuning (not RLHF) |
| **BitDelta** | Per-user personalization | 10× vs LoRA | Similar | Millions of user variants |
| **DeltaSoup** | Community learning | Efficient | Async | Aggregate user improvements |

**Key Distinctions**:
- **Training-Free GRPO**: For RLHF (no value network, default in Zoo Gym)
- **PPO**: For RLHF (with value network, traditional approach)
- **LoRA**: For supervised fine-tuning (not RLHF)
- **BitDelta**: For per-user personalization (works on deltas, not RLHF)
- **DeltaSoup**: For community aggregation (Byzantine-robust, not RLHF)

## Integration with Hanzo Ecosystem

### Python SDK

```python
from hanzo import Hanzo
from zoo.gym import GRPOConfig

hanzo = Hanzo(inference_mode='local')

# Train with GRPO
hanzo.train_grpo(
    base_model="qwen3-4b",
    dataset=preference_dataset,
    config=GRPOConfig(group_size=5, max_epochs=3)
)

# Serve trained model
response = hanzo.chat.completions.create(
    model="qwen3-4b-grpo",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Go SDK

```go
package main

import (
    "context"
    "github.com/hanzoai/go-sdk"
    "github.com/hanzoai/go-sdk/option"
)

func main() {
    client := hanzoai.NewClient(
        option.WithInferenceMode("local"),
    )

    // Train with GRPO
    job, _ := client.Training.GRPO.Create(
        context.Background(),
        hanzoai.GRPOTrainingParams{
            BaseModel: hanzoai.F("qwen3-4b"),
            Dataset: hanzoai.F(preference_dataset),
            GroupSize: hanzoai.F(5),
            MaxEpochs: hanzoai.F(3),
        },
    )

    println("Training job:", job.ID)
}
```

### Hanzo Node

```bash
# Start Hanzo Node with GRPO support
hanzo-node start --enable-grpo

# Train with GRPO
hanzo-node train grpo \
    --base-model qwen3-4b \
    --dataset ./preference_data.jsonl \
    --group-size 5 \
    --max-epochs 3 \
    --output ./trained_model

# Serve trained model
hanzo-node serve --model ./trained_model
```

### Hanzo Engine

Training-Free GRPO is optimized in Hanzo Engine (Rust):
- Native Rust implementation
- Zero-copy experience injection
- Memory-efficient group sampling
- Multi-query parallel GRPO

### Hanzo Dev

```bash
# Train with CLI
dev "train GRPO on math dataset with 100 samples, group size 5"

# Export trained model
dev "export GRPO model to GGUF format"

# Deploy to local node
dev "deploy GRPO model to hanzo node"
```

### Hanzo Desktop

- Train GRPO on MacBook (M1/M2/M3)
- Mine $AI tokens while training
- ZenLM models (zen-eco, zen-agent) with GRPO
- Local RLHF for privacy

## Best Practices

### 1. Group Size Selection

```python
# Small datasets (< 100 samples): group_size=3-5
# Medium datasets (100-1000): group_size=5-8 (default)
# Large datasets (> 1000): group_size=8-10
```

### 2. Temperature Tuning

```python
# Classification tasks: temperature_train=0.5, temperature_eval=0.1
# Generation tasks: temperature_train=0.9, temperature_eval=0.7
# Math/reasoning: temperature_train=0.7, temperature_eval=0.3 (default)
```

### 3. Sample Efficiency

```python
# Training-Free GRPO works with 100-1000 samples
# More samples = better but diminishing returns
# 100 samples: ~80% of max performance
# 1000 samples: ~95% of max performance
```

## Limitations and Considerations

### 1. Group Size Trade-off
- Larger groups = more stable but slower
- Smaller groups = faster but less stable
- Recommended: 5-8 for most tasks

### 2. Sample Efficiency
- Training-Free GRPO needs 100-1000 samples
- Still orders of magnitude fewer than PPO/LoRA
- Quality > Quantity for GRPO

### 3. Base Model Quality
- Training-Free GRPO relies on frozen base model
- Larger base models generally work better
- Example: 671B DeepSeek outperforms fine-tuned 32B

## Related Skills

- **bitdelta.md** - Per-user personalization (10× compression, distinct from GRPO)
- **deltasoup.md** - Community-driven learning (Byzantine-robust aggregation)
- **../hanzo/hanzo-gym.md** - Comprehensive Zoo Gym training guide
- **../hanzo/python-sdk.md** - Python SDK integration
- **../hanzo/go-sdk.md** - Go SDK integration
- **../hanzo/hanzo-node.md** - Local inference infrastructure
- **../hanzo/hanzo-engine.md** - Native Rust inference engine

## Additional Resources

- **GitHub**: https://github.com/zooai/gym
- **Paper**: https://arxiv.org/abs/2502.01155 (DeepSeek GRPO)
- **Zoo Labs Foundation**: https://zoo.ngo (501(c)(3) wildlife conservation)

---

**Remember**: Training-Free GRPO is **Zoo Gym's default RLHF method** with 40% memory reduction, 2× speedup, and 500× cost savings compared to traditional RL - perfect for RLHF with minimal samples and no value network overhead.
