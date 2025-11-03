import os
import yaml
from collections import namedtuple

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator

from model import HuggingFaceLM
from datasets import GSM8KDataset, MathExprDataset
from reward import reward_function
from grpo_loss import PolicyLoss


# Set precision for matrix multiplications
torch.set_float32_matmul_precision('medium')

# Open and load the YAML file as a dictionary
with open("grpo_config.yml", 'r') as file:
    config = yaml.safe_load(file)

# Get training data (Replace with custom data) - using batch_size from config
train_loader = DataLoader(GSM8KDataset(split="train"), batch_size=config['batch_size'], pin_memory=True, prefetch_factor=4, num_workers=2)
# Load test dataset to generate sample answers 
test_dataset = GSM8KDataset(split="test") 

# Edit system prompt if required 
system_prompt = """A conversation between User and Assistant. 
When the user asks to solve a math expression, you must provide a step-by-step reasoning followed by the final answer. 
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
    use_lora=config['use_lora'], 
    lora_config=config.get('lora_config') if config['use_lora'] else None,
    cache_dir=config.get('cache_dir')
)
llm.model.train()

# Initialize reference model (without LoRA)
llm_ref = HuggingFaceLM(
    model_identifier=config['model_identifier'], 
    system_prompt=system_prompt,
    cache_dir=config.get('cache_dir')
)
llm_ref.model.eval()

# Initialize training components (loss, optimizer)
loss_fn = PolicyLoss(kl_weight=float(config['kl_weight']))
optimizer = AdamW(llm.model.parameters(), lr=float(config['learning_rate']))

# Prepare models, optimizer, and data loader with Accelerator
accelerator = Accelerator()
llm.model, llm_ref.model, optimizer, train_loader = accelerator.prepare(llm.model, llm_ref.model, optimizer, train_loader)

# Define the Experience tuple for storing rollout data
Experience = namedtuple('Experience', [
    'sequence_ids',    # Token IDs for generated sequences
    'old_log_probs',   # Log probabilities from current policy
    'ref_log_probs',   # Log probabilities from reference model
    'advantages',      # Advantage values for each sequence
    'action_mask'      # Mask for valid actions
])

# Training loop variables
steps = 0
total_reward = 0
exps = []

# Create checkpoint directory
os.makedirs(config['checkpoint_dir'], exist_ok=True)
os.makedirs(config['log_dir'], exist_ok=True)

# Create tensorboard writer
writer = SummaryWriter(config['log_dir'])

print("---------------------------------------------------------")
print("GRPO Training started...")
print(f"LoRA Enabled: {config['use_lora']}")
print(f"Batch Size: {config['batch_size']}")
print(f"Group Size: {config['group_size']}")
print(f"Learning Rate: {config['learning_rate']}")
print(f"KL Weight: {config['kl_weight']}")
print(f"Rollouts per Step: {config['rollouts_per_step']}")
print(f"Updates per Step: {config['updates_per_step']}")
print("---------------------------------------------------------")

for batch in train_loader:
    steps += 1
    
    with torch.no_grad():
        # Repeat each prompt in the batch by group_size
        all_prompts = []
        all_targets = []
        
        for prompt, target in zip(batch['prompt'], batch['answer']):
            all_prompts.extend([prompt] * config['group_size'])
            all_targets.extend([target] * config['group_size'])
        
        # Generate sequences for all prompts (batch_size * group_size)
        sequence_ids, completions, action_masks = llm.generate(all_prompts, max_length=config['max_completion_len'])

        # Compute log probabilities
        log_probs = llm.compute_log_probs(sequence_ids)
        ref_log_probs = llm_ref.compute_log_probs(sequence_ids)
        
        # Compute rewards for all completions
        rewards = torch.tensor([reward_function(target, completion) for target, completion in zip(all_targets, completions)],
                                dtype=torch.float, device=accelerator.device)

        # Reshape rewards to (batch_size, group_size) for advantage calculation
        rewards_reshaped = rewards.view(config['batch_size'], config['group_size'])
        
        # Compute advantages per group
        advantages_list = []
        batch_mean_rewards = []
        
        for i in range(config['batch_size']):
            group_rewards = rewards_reshaped[i]
            mean_r = group_rewards.mean()
            std_r = group_rewards.std()
            
            # Normalize advantages within each group
            if std_r > 1e-8:
                group_advantages = (group_rewards - mean_r) / (std_r + 1e-8)
            else:
                group_advantages = group_rewards - mean_r
            
            advantages_list.append(group_advantages)
            batch_mean_rewards.append(mean_r)
        
        # Concatenate all advantages back to flat tensor
        advantages = torch.cat(advantages_list)
        
        # Calculate average reward for this batch
        batch_avg_reward = torch.stack(batch_mean_rewards).mean()
        total_reward += batch_avg_reward

        # Store experience
        exp = Experience(
            sequence_ids=sequence_ids.detach(),
            old_log_probs=log_probs.detach(),
            ref_log_probs=ref_log_probs.detach(),
            advantages=advantages.detach(),
            action_mask=action_masks.detach()
        )
        exps.append(exp)
    
    torch.cuda.empty_cache()

    # Log and update model periodically
    if steps % config['rollouts_per_step'] == 0:
        avg_reward = total_reward / config['rollouts_per_step']
        writer.add_scalar('Reward', avg_reward, steps)
        print(f"Step: {steps}  Avg Reward: {avg_reward.item():.4f}")
        total_reward = 0

        # Multiple updates per batch of experiences
        for _ in range(config['updates_per_step']):
            for exp in exps:
                optimizer.zero_grad()
                log_probs = llm.compute_log_probs(exp.sequence_ids)
                loss = loss_fn(log_probs, 
                                exp.old_log_probs, 
                                exp.advantages, 
                                exp.ref_log_probs, 
                                exp.action_mask)
                accelerator.backward(loss)
                torch.nn.utils.clip_grad_norm_(llm.model.parameters(), max_norm=1.0)
                optimizer.step()
        
        exps = []
        torch.cuda.empty_cache()

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
final_path = os.path.join(config['checkpoint_dir'], 'model_grpo_final')
llm.model.save_pretrained(final_path)
print(f"Final checkpoint saved to {final_path}")
writer.close()

