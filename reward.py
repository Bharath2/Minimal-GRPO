import re
import numpy as np

def reward_function(target, completion):
    """Calculate reward for generated sequence compared to the completion.
    
    Args:
        target: Target answer string 
        completion: Generated completion string containing answer
        
    Returns:
        float: Reward value between 0 and 5.
    """
    # Only one <think> and <answer> tag should be present
    if any(completion.count(tag) != 1 for tag in ["<think>", "</think>", "<answer>", "</answer>"]):
        return 0.0

    # Extract answer from completion text between <answer> tags
    pattern = r"<think>\n(.*?)\n</think>\n<answer>(.*?)</answer>"
    match = re.search(pattern, completion, flags=re.DOTALL)
    if match: generated_answer = match.group(2)
    else: return 0.0
    
    if target.isdigit():
        # Handle numeric answers 
        target_num = int(target)
        if generated_answer.isdigit():
            generated_num = int(generated_answer)
        else: 
            return 0.0
        value_diff = abs(target_num - generated_num) 
        if value_diff == 0:
            base_reward = 5.0 # Exact match reward
        else:   
            base_reward = 3.0 * np.clip(1 - value_diff/10, 0, 1) # Close match reward
        return base_reward + 1.0
    else:
        # Handle algebraic expressions
        generated_answer = generated_answer.replace('^', '**')
        # Split expressions into terms
        target_terms = set(term.strip() for term in re.split(r'[+\-]', target) if term.strip())
        generated_terms = set(term.strip() for term in re.split(r'[+\-]', generated_answer) if term.strip())
        # Perfect match
        if target_terms == generated_terms:
            return 5.0  # Exact match reward
        # Partial match based on overlapping terms
        matching_terms = len(target_terms.intersection(generated_terms))
        return min(3.0, matching_terms * 1.0) + 1.0
