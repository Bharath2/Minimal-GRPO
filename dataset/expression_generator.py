import random
import re

class ExpressionGenerator:
    """
    A class to generate random mathematical expressions with controlled complexity.
    
    The generator creates expressions using integers, basic operators (+, -, *), and parentheses. 

    Examples:
       (30+18-((24)+1))+92-(((93))) = 22
       ((54-27)*(18))+88 = 574
    """

    def __init__(self, max_length=64, max_int=100, max_depth=4, max_ops=3):
        self.max_depth = max_depth
        self.max_length = max_length
        self.max_ops = max_ops
        self.max_int = max_int

    def generate_expr(self, max_depth, current_depth, ops_remaining):
        """
        Generate an expression with at most ops_remaining operations.
        It starts with a factor then, while ops_remaining > 0 and randomly, adds an operator
        (from '+', '-', '*') and another factor. Each added operator decrements ops_remaining.
        """
        expr = self.generate_factor(max_depth, current_depth)
        while ops_remaining > 0 and random.random() < 0.5 and len(expr) < self.max_length:
            op = random.choice(["+", "-", "*"])
            expr += op + self.generate_factor(max_depth, current_depth)
            ops_remaining -= 1
        return expr

    def generate_factor(self, max_depth, current_depth):
        """
        Generate a factor: either a simple integer or a parenthesized expression.
        When generating a parenthesized expression, the allowed operations counter resets.
        """
        if current_depth >= self.max_depth or random.random() < 0.5:
            #Generate a random integer between 0 and max_int
            return str(random.randint(0, self.max_int))
        else:
            # Generate a subexpression (with at most max_ops operations) inside parentheses.
            return "(" + self.generate_expr(max_depth, current_depth + 1, self.max_ops) + ")"

    def __call__(self):
        """
        Generate a math expression and its evaluated result.
        """
        max_depth = random.randint(1, self.max_depth)
        expr = self.generate_expr(max_depth, 0, self.max_ops)
        return expr, eval(expr)

if __name__ == "__main__":
    # Test the ExpressionGenerator with different configurations
    generators = [
        ExpressionGenerator(max_depth=2, max_ops=1),
        ExpressionGenerator(max_depth=3, max_ops=2),
        ExpressionGenerator(max_depth=4, max_ops=3)
    ]
    
    print("Testing expression generation")
    for gen in generators:
        print(f"\nGenerator (max_depth={gen.max_depth}, max_ops={gen.max_ops}):")
        for _ in range(5):
            expr, result = gen()
            print(f"{expr} = {result}")
