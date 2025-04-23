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
        return -0.25

    # Extract answer from completion text between <answer> tags
    pattern = r"^\s*<think>\n(.*?)\n</think>\n<answer>(.*?)</answer>\s*$"
    match = re.search(pattern, completion, flags=re.DOTALL)
    if match: 
        generated_answer = match.group(2)
    else: 
        return -0.25
        
    try:
        if target.isdigit():
            # Handle numeric answers 
            target_num = int(target)
            if generated_answer.isdigit():
                generated_num = int(generated_answer)
            else: return -0.25
            value_diff = abs(target_num - generated_num) 
            # Exact match reward   
            if value_diff == 0: return 1.0 
            #Close match reward
            reward = 3.0 * np.clip(1 - value_diff/10, 0, 1) 
            return (reward + 1.0)/5.0
        else:
            # Handle algebraic expressions
            generated_answer = generated_answer.replace('^', '**')
            # Split expressions into terms
            target_terms = set(term.strip() for term in re.split(r'[+\-]', target) if term.strip())
            generated_terms = set(term.strip() for term in re.split(r'[+\-]', generated_answer) if term.strip())
            # Perfect match reward
            if target_terms == generated_terms:
                return 1.0 
            # Partial match based on overlapping terms
            matching_terms = len(target_terms.intersection(generated_terms))
            reward = min(3.0, matching_terms)
            return (reward + 1.0)/5.0
    except: 
        return -0.25
