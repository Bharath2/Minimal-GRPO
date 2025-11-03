import os
import yaml
import random
from tqdm import tqdm

import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator

from model import HuggingFaceLM
from datasets import GSM8KDataset, MathExprDataset
from reward import reward_function


# Set precision for matrix multiplications
torch.set_float32_matmul_precision('medium')

# Open and load the YAML file as a dictionary
with open("es_config.yml", 'r') as file:
    config = yaml.safe_load(file)

# Get training data (Replace with custom data)
train_loader = DataLoader(GSM8KDataset(split="train"), batch_size=config['prompts_per_iteration'], num_workers=4, pin_memory=True)
# Load test dataset to generate sample answers 
test_dataset = GSM8KDataset(split="test") 

# Edit system prompt if required 
system_prompt = """A conversation between User and Assistant. 
When the user asks to solve a math problem, you must provide a step-by-step reasoning followed by the final answer. 
your intermediate steps must be enclosed in <think>...</think> tag, and the final answer enclosed in <answer>...</answer> tag. 
final answer must be plain text with no units or symbols (ex: 24, not '$24' or '= 24').
Each step must be numbered as shown below:
<think>
1. The first step here
2. The second step here
and so on...
</think>
<answer>final answer</answer>
"""

# Initialize the model (to fine-tune)
llm = HuggingFaceLM(
    model_identifier=config['model_identifier'], 
    system_prompt=system_prompt, 
    cache_dir=config.get('cache_dir')
)
llm.model.eval()  # ES doesn't use training mode (no dropout during evaluation)

# Prepare model and data loader with Accelerator
accelerator = Accelerator()
llm.model, train_loader = accelerator.prepare(llm.model, train_loader)


# Helper functions for ES
def get_trainable_params_list(model):
    """Get list of trainable parameters."""
    trainable_params = []
    for param in model.parameters():
        if param.requires_grad:
            trainable_params.append(param)
    return trainable_params


def perturb_params_with_seed(params_list, seed, noise_scale):
    """
    Perturb parameters in-place using a random seed.
    This allows us to regenerate the same noise later without storing it.
    """
    # Set the random seed for reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    # Perturb each parameter
    for param in params_list:
        noise = torch.randn_like(param.data)
        param.data.add_(noise, alpha=noise_scale)


def restore_params_with_seed(params_list, seed, noise_scale):
    """
    Restore parameters in-place by regenerating the same noise and subtracting.
    """
    # Set the same random seed to regenerate identical noise
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    # Restore each parameter
    for param in params_list:
        noise = torch.randn_like(param.data)
        param.data.sub_(noise, alpha=noise_scale)


def update_params_with_seed(params_list, seed, z_score, learning_rate, population_size):
    """
    Update parameters in-place using regenerated noise and z-score.
    Update rule: theta_t = theta_(t-1) + learning_rate * (1/N) * Z_n * noise_n
    """
    # Set the same random seed to regenerate identical noise
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    # Update each parameter
    update_coef = learning_rate * z_score / population_size
    for param in params_list:
        noise = torch.randn_like(param.data)
        param.data.add_(noise, alpha=update_coef)


# Create checkpoint directory
os.makedirs(config['checkpoint_dir'], exist_ok=True)
os.makedirs(config['log_dir'], exist_ok=True)

# Create tensorboard writer
writer = SummaryWriter(config['log_dir'])

print("---------------------------------------------------------")
print("Evolutionary Strategy Training started...")
print(f"Population Size: {config['population_size']}")
print(f"Noise Scale (sigma): {config['noise_scale']}")
print(f"Learning Rate (alpha): {config['learning_rate']}")
print(f"Prompts per Iteration: {config['prompts_per_iteration']}")
print("---------------------------------------------------------")

# Get trainable parameters 
trainable_params = get_trainable_params_list(llm.model)

for iteration, batch in enumerate(train_loader):
    if iteration > config['total_iterations']: break
    
    print(f"\nIteration: {iteration}/{config['total_iterations']}")
    
    # Sample prompts for this iteration
    num_prompts = min(config['prompts_per_iteration'], len(batch['prompt']))
    prompts = list(batch['prompt'][:num_prompts])
    targets = list(batch['answer'][:num_prompts])
    
    # Sample N random seeds s1, s2, ..., sN
    random_seeds = [random.randint(0, 2**30) for _ in range(config['population_size'])]
    
    rewards = []

    for seed in tqdm(random_seeds, miniters=4, mininterval=0.0):
        # Perturb parameters with noise
        perturb_params_with_seed(trainable_params, seed, config["noise_scale"])

        # Evaluate model with greedy decoding (deterministic)
        with torch.no_grad():
            # Batch generate all prompts at once using greedy decoding
            _, completions, _ = llm.generate(prompts, max_length=config["max_completion_len"], do_sample=False)
            # Compute rewards for each prompt
            prompt_rewards = [reward_function(target, completion) for target, completion in zip(targets, completions)]
            avg_reward = sum(prompt_rewards) / len(prompts)

        # Restore parameters to original state
        restore_params_with_seed(trainable_params, seed, config["noise_scale"])
        rewards.append(avg_reward)
    
    # Convert rewards to numpy for statistics
    rewards_np = np.array(rewards)
    
    # Normalize rewards: Z_n = (R_n - R_mean) / R_std (z-score normalization)
    mean_reward = rewards_np.mean()
    std_reward = rewards_np.std()
    max_reward = rewards_np.max()
    min_reward = rewards_np.min()
    
    if std_reward > 1e-6:
        z_scores = (rewards_np - mean_reward) / (std_reward)
    else:
        z_scores = rewards_np - mean_reward
    
    # Update parameters using z-scores (in main process)
    # Update rule: theta_t = theta_(t-1) + alpha * (1/N) * sum(Z_n * noise_n)
    for seed_n, z_n in zip(random_seeds, z_scores):
        # Update in-place using regenerated noise
        update_params_with_seed(trainable_params, seed_n, z_n, 
                                config['learning_rate'], config['population_size'])
        

    # Log to tensorboard
    writer.add_scalar('Reward/Mean', mean_reward, iteration)
    writer.add_scalar('Reward/Max', max_reward, iteration)
    writer.add_scalar('Reward/Min', min_reward, iteration)
    writer.add_scalar('Reward/Std', std_reward, iteration)
    
    print(f"Avg Reward: {mean_reward:.4f}  Max: {max_reward:.4f}  Min: {min_reward:.4f} Std: {std_reward:.4f}")
    
    # Save checkpoint periodically
    if iteration % config['save_every'] == 0:
        checkpoint_path = os.path.join(config['checkpoint_dir'], f'model_es_iter_{iteration}')
        print(f"Saving checkpoint to {checkpoint_path}...")
        # llm.model.save_pretrained(checkpoint_path)
        
    # Sample random test examples to evaluate model
    with torch.no_grad():
        test_samples = test_dataset.sample(n=2)
        test_prompts = [sample['prompt'] for sample in test_samples]
        test_targets = [sample['answer'] for sample in test_samples]
        
        _, completions, _ = llm.generate(test_prompts, max_new_tokens=config["max_completion_len"], do_sample=False)
        for prompt, target, generated in zip(test_prompts, test_targets, completions):
            print('---------------------------------------')
            print(f"Prompt: {prompt}")
            print(f"Target: {target}")
            print(f"Generated: {generated}")
            print('---------------------------------------')

    torch.cuda.empty_cache()

# Save final model
print("\nTraining completed. Saving final checkpoint...")
final_path = os.path.join(config['checkpoint_dir'], 'model_es_final')
llm.model.save_pretrained(final_path)
print(f"Final checkpoint saved to {final_path}")
writer.close()

