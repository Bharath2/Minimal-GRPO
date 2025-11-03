# Minimal-GRPO

A minimal and hackable implementation of fine-tuning Open Language Models (LLaMA, Qwen, etc.) on mathematical reasoning tasks using two approaches:

- **GRPO (Group Relative Policy Optimization)**: Gradient-based RL policy optimization with support for LoRA adaptation
- **ES (Evolution Strategies)**: Gradient-free evolutionary optimization with full parameter fine-tuning

Both approaches are designed to be simple, customizable, and effective for reinforcement learning from verifiable rewards. This repo currently includes **GSM8K** and **MathExpr** datasets, and can be easily adapted to other datasets and tasks.

### GRPO (Group Relative Policy Optimization)

Based on the approach from [DeepSeekMath](https://arxiv.org/abs/2402.03300), GRPO is a gradient-based reinforcement learning algorithm that:
- Uses group-based advantage estimation for stable policy updates
- Combines PPO-style clipping with KL divergence regularization
- Maintains a reference model to prevent catastrophic forgetting
- Efficiently leverages gradient information for policy improvement
- **Uses LoRA (Low-Rank Adaptation)** for parameter-efficient fine-tuning

### ES (Evolution Strategies)

Based on recent work showing ES can scale to billion-parameter LLMs ([Qiu et al., 2025](https://arxiv.org/pdf/2509.24372)), Evolution Strategies is a gradient-free optimization approach inspired by natural evolution. It:
- Perturbs model parameters with Gaussian noise
- Evaluates fitness using task-specific rewards
- Updates parameters based on relative performance (z-score normalization)
- Requires only forward passes (no backpropagation)
- Less prone to reward hacking than RL methods

Recent research demonstrates that ES can successfully fine-tune LLMs with billions of parameters, outperforming RL methods in sample efficiency, robustness, and stability, particularly on tasks with sparse outcome-only rewards.

## Features

- ✅ **Clean, minimal codebase**: Easy to understand and modify
- ✅ **Two optimization strategies**: Compare gradient-based (GRPO) vs gradient-free (ES) approaches
- ✅ **LoRA support for GRPO**: Parameter-efficient fine-tuning with PEFT
- ✅ **Full parameter ES fine-tuning**: Direct optimization in billion-parameter spaces
- ✅ **Flexible configuration**: YAML-based configuration for both algorithms
- ✅ **Multiple datasets**: GSM8K and MathExpr included, easily extensible
- ✅ **Custom rewards**: Easily adapt to your own tasks and reward functions

## Requirements

**Core Dependencies:**
```
pytorch
transformers
peft
pyyaml
tensorboard
accelerate
numpy
```
**Optional (recommended for efficiency):**
```
flash-attn
```
## Quick Start

### GRPO Training

1. Configure your training in `grpo_config.yml`

2. Run training:
```bash
python grpo_train.py
```

3. Monitor with TensorBoard:
```bash
tensorboard --logdir=grpo_logs
```

### ES Training

1. Configure your training in `es_config.yml`

2. Run training:
```bash
python es_train.py
```

3. Monitor with TensorBoard:
```bash
tensorboard --logdir=es_logs
```

## Datasets and Custom Tasks

### Included Datasets

The project currently includes two mathematical reasoning datasets:

1. **GSM8KDataset**: Grade School Math 8K problems - a dataset of grade school math word problems
2. **MathExprDataset**: Mathematical expression evaluation and manipulation tasks

Both datasets are implemented in the `datasets.py` file and can be easily swapped in the training scripts.

### Adapting to Your Own Tasks

To adapt this code to your own dataset and task:

1. **Implement your dataset** in `datasets.py` (follow the GSM8K or MathExpr examples)
2. **Define your reward function** in `reward.py` to match your task's success criteria
3. **Adjust the system prompt** in the training scripts (`grpo_train.py` or `es_train.py`) to match your task format
4. **Update the DataLoader** in the training script to use your new dataset

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

See [LICENSE](LICENSE) file for details.

## References

### GRPO
- **DeepSeekMath**: Shao et al., "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models", 2024. [arXiv:2402.03300](https://arxiv.org/abs/2402.03300)

### Evolution Strategies
- **ES at Scale for LLMs**: Qiu et al., "Evolution Strategies at Scale: LLM Fine-Tuning Beyond Reinforcement Learning", 2025. [arXiv:2509.24372](https://arxiv.org/pdf/2509.24372)
- Salimans et al., "Evolution Strategies as a Scalable Alternative to Reinforcement Learning", 2017. [arXiv:1703.03864](https://arxiv.org/abs/1703.03864)

## Acknowledgments

- Built with [PyTorch](https://pytorch.org/), [Transformers](https://huggingface.co/transformers/), and [Accelerate](https://huggingface.co/docs/accelerate/)
- Models from [Hugging Face](https://huggingface.co/)
