import re
import numpy as np

# number tokens: int/float, optional sign, optional scientific notation
NUM_RE   = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
PLAIN_RE = re.compile(r"^\s*[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?\s*$")

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
        return -0.3

    # Extract answer from completion text between <answer> tags
    pattern = r"^\s*<think>\n(.*?)\n</think>\n<answer>(.*?)</answer>\s*$"
    match = re.search(pattern, completion, flags=re.DOTALL)
    if match:  generated_answer = match.group(2)
    else: return -0.3
        
    try:
        tgt_str = str(target).strip().replace(",", "")
        if NUM_RE.fullmatch(tgt_str):
            target_num = float(tgt_str)
            # Find first numeric token in generated answer
            gen_search = NUM_RE.search(generated_answer.replace(",", ""))
            if not gen_search: return -0.2
            alpha = 1.0 if PLAIN_RE.fullmatch(generated_answer) else 0.9
            generated_num = float(gen_search.group(0))
            rel_err = abs(target_num - generated_num) / max(1e-12, abs(target_num))
            # close match, full if plain number only, else slightly reduced
            if rel_err <= 1e-6: return alpha
            # Close match reward (until 10% of error)
            closeness = np.clip(1.0 - rel_err / 0.10, 0.0, 1.0)
            return alpha * (2.5 * closeness + 1.0) / 5.0
        else:
            # Handle algebraic expressions
            generated_answer = generated_answer.replace('^', '**')
            # Split expressions into terms
            target_terms = set(term.strip() for term in re.split(r'[+\-]', target) if term.strip())
            generated_terms = set(term.strip() for term in re.split(r'[+\-]', generated_answer) if term.strip())
            # Perfect match reward
            if target_terms == generated_terms: return 1.0 
            # Partial match based on overlapping terms
            matching_terms = len(target_terms.intersection(generated_terms))
            closeness = matching_terms / max(1, len(target_terms))
            return (3.0 * closeness + 1.0) / 5.0
    except: 
        return -0.25