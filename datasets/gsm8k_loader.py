import os, json, re, random
import torch.utils.data as data

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"

def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]

def extract_answer(text: str):
    m = ANS_RE.search(text)
    if not m:
        return INVALID_ANS
    s = m.group(1).strip().replace(",", "")
    return s

class GSM8KDataset(data.Dataset):
    """
    Returns raw prompt/answer pairs suitable for generative training/eval.

    Each item:
      {
        'prompt':  "<question text>",
        'answer':  "<direct final answer string>"
      }
    """
    def __init__(
        self,
        split: str,
        data_dir: str|None = None,
        skip_invalid: bool = True, # skip examples where final answer can't be parsed
        shuffle: bool = True
    ):
        # Try to find gsm8k_data folder relative to this file if data_dir is not present
        if data_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            alt_data_dir = os.path.join(script_dir, "gsm8k_data")
            if not os.path.exists(alt_data_dir):
                raise FileNotFoundError(f"gsm8k_data folder not found at {data_dir} or {alt_data_dir}")
            data_dir = alt_data_dir
            
        path = os.path.join(data_dir, f"{split}.jsonl")
        raw = read_jsonl(path)

        examples = []
        for ex in raw:
            # Normalize fields
            q = (ex["question"] or "").rstrip()  # no trailing newline by default
            a = extract_answer(ex["answer"] or "")

            if a == INVALID_ANS and skip_invalid:
                continue
            
            examples.append({"prompt": q, "answer": a})

        if shuffle: random.shuffle(examples)

        self.examples = examples
        print(f"{len(self.examples)} {split} examples (plain prompt/answer)")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
    
    def sample(self, n=2):
        """
        Sample n random examples from the dataset.
        
        Args:
            n: Number of examples to sample (default: 2)
            
        Returns:
            List of dictionaries with 'prompt' and 'answer' keys
        """
        return random.sample(self.examples, min(n, len(self.examples)))