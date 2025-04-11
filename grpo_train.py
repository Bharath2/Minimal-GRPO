import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim import AdamW
from collections import namedtuple
import os
from accelerate import Accelerator

from model import HuggingFaceLM
from dataset import MathDataset
from reward import reward_function
from loss import PolicyLoss
import yaml

# Set precision for matrix multiplications
torch.set_float32_matmul_precision('medium')

# Open and load the YAML file as a dictionary
with open("config.yml", 'r') as file:
    config = yaml.safe_load(file)
    
system_prompt = """A conversation between User and Assistant. 
When the user asks to solve a math expression, you must provide a step-by-step reasoning followed by the final answer. 
your intermediate steps must be enclosed in <think>...</think> tag, and the final answer enclosed in <answer>...</answer> tag. 
Each step must be numbered as shown below:
<think>
1. The first step here
2. The second step here
and so on...
</think>
<answer>final answer</answer>
"""

# Initialize Accelerator
accelerator = Accelerator()

# current model
llm = HuggingFaceLM(model_identifier=config['model_identifier'], system_prompt=system_prompt, 
                    use_lora=config['use_lora'], lora_config=config['lora_config'])
llm.model.train()

# Reference model
llm_ref = HuggingFaceLM(model_identifier=config['model_identifier'], system_prompt=system_prompt)
llm_ref.model.eval()

# Get training data
train_loader = DataLoader(MathDataset(), batch_size=256, num_workers=4, pin_memory=True)

# Initialize training components
loss_fn = PolicyLoss(kl_weight=float(config['kl_weight']))
optimizer = AdamW(llm.model.parameters(), lr=float(config['learning_rate']))

# Prepare models, optimizer, and data loader with Accelerator
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
reward = 0
exps = []
writer = SummaryWriter(config['log_dir'])

print("---------------------------------------------------------")
print("Training started...")

for batch in train_loader:
    for prompt, target in zip(batch['prompt'], batch['answer']):
        steps += 1
        with torch.no_grad():
            # Generate multiple sequences for each prompt
            prompts = [prompt] * config['group_size']
            sequence_ids, completions, action_masks = llm.generate(prompts, max_length=config['max_completion_len'])

            # Compute log probabilities and rewards
            log_probs = llm.compute_log_probs(sequence_ids)
            ref_log_probs = llm_ref.compute_log_probs(sequence_ids)
            rewards = torch.tensor([reward_function(target, completion) for completion in completions],
                                    dtype=torch.float, device=accelerator.device)

            # Compute advantages
            mean_r = rewards.mean()
            std_r = rewards.std()
            advantages = (rewards - mean_r) # / (std_r + 1e-6)
            reward += mean_r

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

        # Log and Update model periodically
        if steps % config['rollouts_per_step'] == 0:
            avg_reward = reward/config['rollouts_per_step']
            writer.add_scalar('Reward', avg_reward, steps)
            print("step:", steps, "  Avg Reward:", avg_reward.item())
            reward = 0

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

            # Generate test samples
            with torch.no_grad():
                test_expressions = [
                    "Expand 3*(x+1)*(x+2)",
                    "Solve 78-(3*(47-2)+2)"
                ]
                _, completions, _ = llm.generate(test_expressions, top_p=0.5)
                for prompt, answer in zip(test_expressions, completions):
                    print('---------------------------------------')
                    print(prompt)
                    print(answer)
                    print('---------------------------------------')
            torch.cuda.empty_cache()
            
# Save final model
print("Training completed. Saving final checkpoint...")
llm.model.save_pretrained(os.path.join(config['checkpoint_dir'], 'model_grpo_final'))
print("Final checkpoint saved successfully.")
