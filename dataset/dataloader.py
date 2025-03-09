import re
import torch
import random
from torch.utils.data import Dataset, DataLoader
from .expression_generator import *
import sympy as sp

class PolynomialData:
    def __init__(self, file_path):
        """
        Loads and processes polynomial expressions from a file.
        
        Args:
            file_path: Path to the text file containing one polynomial expression per line.
        """
        self.raw_data = []
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.raw_data.append(line)
                    
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


class MathDataset(Dataset):
    def __init__(self, file_path = './dataset/poly.txt', max_len=64):
        """
        Initializes the dataset.
        
        Args:
          file_path: Path to the text file containing one polynomial expression per line.
          max_len: Maximum length for tokenized sequences.
        """
        self.max_len = max_len
        self.poly_data = PolynomialData(file_path)
        self.expr_data = ExpressionGenerator(max_length=max_len)
    
    def __len__(self):
        return len(self.poly_data)
    
    def __getitem__(self, idx):
        # With 50% probability, generate a new expression
        if random.random() < 0.5:
            prompt, answer = self.expr_data()
            prefix = random.choice(["What is ", "Solve ", ""])
            prompt = prefix + prompt
        else:
            prompt, answer = self.poly_data.sample()
            prefix = random.choice(["Expand ", "Solve ", ""])
            prompt = prefix + prompt
        return {'prompt': str(prompt), 'answer': str(answer)}


if __name__ == "__main__":    
    # Use actual path to text file
    file_path = "./poly.txt"
    batch_size = 4  # Change as needed
    max_len = 64    # Fixed length of sequences
    
    # Create DataLoader using the MathDataset
    dataloader = DataLoader(
        MathDataset(file_path),
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
