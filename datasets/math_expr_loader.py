import os
import random
from torch.utils.data import Dataset, DataLoader
from .math_expr_data import *
import sympy as sp

class PolynomialData:
    def __init__(self, file_path, split='train'):
        """
        Loads and processes polynomial expressions from a file.
        
        Args:
            file_path: Path to the text file containing one polynomial expression per line.
            split: Either 'train' or 'test'. Train uses first 70%, test uses remaining 30%.
        """
        self.raw_data = []
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.raw_data.append(line)
        
        # Apply train/test split
        split_idx = int(len(self.raw_data) * 0.7)
        if split == 'train':
            self.raw_data = self.raw_data[:split_idx]
        elif split == 'test':
            self.raw_data = self.raw_data[split_idx:]
        else:
            raise ValueError(f"split must be 'train' or 'test', got {split}")
                    
    def __len__(self):
        return len(self.raw_data)
        
    def sympy_swap_mul(self, expression):
        """
        Uses sympy to parse the expression, then randomly selects one multiplication (Mul)
        node and swaps its operands. Returns a new expression string.
        """
        try:
            expr = sp.sympify(expression, evaluate=False)
            # Find all multiplication nodes and choose one randomly
            mul_nodes = [node for node in sp.preorder_traversal(expr) if isinstance(node, sp.Mul)]
            if not mul_nodes: return expression
            chosen = random.choice(mul_nodes)
            # Create a new multiplication expression with reversed operands
            args = list(chosen.args)
            new_mul = sp.Mul(*args[::-1], evaluate=False)
            new_expr = expr.xreplace({chosen: new_mul})
            return str(new_expr)
        except:
            return expression
        
    def sample(self):
        """Get random expression and answer"""
        idx = random.randrange(len(self.raw_data))
        line = self.raw_data[idx]
        parts = line.split("=")
        prompt = self.sympy_swap_mul(parts[0])
        return parts[0], parts[1]


class MathExprDataset(Dataset):
    """
    Returns raw prompt/answer pairs for math expression evaluation.
    
    Each item:
      {
        'prompt':  "<math expression>",
        'answer':  "<evaluated result>"
      }
    """
    def __init__(
        self,
        split: str,
        data_dir: str|None = None,
        max_len: int = 64,
        shuffle: bool = True
    ):
        """
        Initializes the dataset.
        
        Args:
          split: Either 'train' or 'test'. Train uses first 70%, test uses remaining 30%.
          data_dir: Directory containing the polynomial data file.
          file_name: Name of the file containing polynomial expressions.
          max_len: Maximum length for generated sequences.
          shuffle: Whether to shuffle examples after loading.
        """
        self.max_len = max_len
        if data_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            alt_data_dir = os.path.join(script_dir, "math_expr_data")
            if not os.path.exists(alt_data_dir):
                raise FileNotFoundError(f"math_expr_data folder not found at {data_dir} or {alt_data_dir}")
            data_dir = alt_data_dir
        
        file_path = os.path.join(data_dir, "poly.txt")
        self.poly_data = PolynomialData(file_path, split=split)
        self.expr_data = ExpressionGenerator(max_length=max_len)
        
        # Pre-generate examples for consistent dataset size
        self.examples = []
        for idx in range(len(self.poly_data)):
            # With 50% probability, generate a new expression
            if random.random() < 0.5:
                prompt, answer = self.expr_data()
                prefix = random.choice(["What is ", "Solve ", ""])
                prompt = prefix + prompt
            else:
                prompt, answer = self.poly_data.sample()
                prefix = random.choice(["Expand ", "Solve ", ""])
                prompt = prefix + prompt
            self.examples.append({'prompt': str(prompt), 'answer': str(answer)})
        
        if shuffle:
            random.shuffle(self.examples)
        
        print(f"{len(self.examples)} {split} examples (math expressions)")
    
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


if __name__ == "__main__":    
    batch_size = 4  # Change as needed
    max_len = 64    # Fixed length of sequences
    
    # Create DataLoader using the MathExprDataset
    train_dataset = MathExprDataset(
        split='train',
        data_dir='.',
        file_name='poly.txt',
        max_len=max_len
    )
    
    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    
    # Example iteration
    for i, batch in enumerate(dataloader):
        print("Prompt:", batch['prompt'][0])
        print("Answer:", batch['answer'][0]) 
        print("=" * 40)
        if i >= 3:  # Print 4 times total
            break
