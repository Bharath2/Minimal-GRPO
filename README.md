# Minimal-GRPO
Minimal implementation of Group Relative Policy Optimization (GRPO) to fine-tune Language Models like LlaMa-3.2, Qwen2 for Math Tasks.


## Overview

In this project, I implemented and compared fine-tuning pipelines for a LLaMA-based model using a GRPO strategy inspired by DeepSeekMath. The training data comprises synthetic math queries with expected solutions. This approach leverages group-based relative policy optimization to improve the model's ability to reason through math problems. The framework is flexible and can be extended to other models and tasks.

## Requirements

- Python 3.8+
- PyTorch 1.10+
- Hugging Face Transformers
- Additional dependencies listed in `requirements.txt`

## Usage
Data Preparation
A synthetic dataset of math queries and corresponding results is included in the data/ directory. To regenerate or update the dataset, use:

bash
Copy
python generate_dataset.py
Fine-Tuning
To fine-tune the LLaMA model using the GRPO-based approach, run:

bash
Copy
python finetune_grpo.py --data data/synthetic_math_dataset.json --model llama-base
This script will train the model and save the best checkpoint in the models/ folder.

## Results
Initial experiments show that the GRPO approach significantly enhances the model's performance in solving math expressions and polynomial expansions. Detailed results and metrics can be found in the results/ directory.

## Contributing
Please open an issue or submit a pull request if you have suggestions or improvements.
