import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
)
# from peft import PeftModel
from peft import get_peft_model, LoraConfig, TaskType

class HuggingFaceLM:
    """
    A wrapper class for Hugging Face causal language models that handles text generation and log probability computation.
    
    Args:
        model_identifier (str): Path or identifier for the pretrained model (e.g., "meta-llama/Llama-3.2-1B-Instruct").
        bf16 (bool): Whether to use bfloat16 precision; defaults to True.
        device (str or torch.device): Device to load the model on (e.g., "cuda" or torch.device("cuda")).
        system_prompt (str, optional): A system prompt to prepend to all inputs.
        model_class (callable, optional): The model class to use; defaults to AutoModelForCausalLM.
    """
    def __init__(
        self,
        model_identifier="meta-llama/Llama-3.2-1B-Instruct",
        bf16=True,
        device=torch.device("cuda"),
        system_prompt=None,
        model_class = None,
        use_lora = False,
        lora_config = None
    ):
        # Initialize tokenizer with padding token
        self.tokenizer = AutoTokenizer.from_pretrained(model_identifier)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        if model_class is None:
            model_class = AutoModelForCausalLM
        
        # Load model with specified configuration
        self.model = model_class.from_pretrained(
            model_identifier,
            trust_remote_code=False,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16 if bf16 else "auto",
            device_map=device,
        )
        self.system_prompt = system_prompt
        
        if use_lora:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=lora_config['r'],
                lora_alpha=lora_config['alpha'],
                lora_dropout=lora_config['dropout'],
                target_modules=lora_config["target_modules"]
            )
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()

        # self.model.config.use_cache = False 
        # self.model.gradient_checkpointing_enable()

        
    def generate(self, tasks, temperature=1.0, top_p=0.75, max_length=640):
        """
        Generate completions for a batch of tasks.
        
        Args:
            tasks (List[str]): List of input prompts
            temperature (float): Sampling temperature. Defaults to 1.0.
            top_k (int): Consider top k tokens. Defaults to 25.
            max_length (int): Maximum sequence length. Defaults to 640.
            
        Returns:
            tuple: (sequence_ids, completions, action_mask) containing:
                - sequence_ids (Tensor): Token IDs for full sequences
                - completions (List[str]): Generated text completions
                - action_mask (Tensor): Boolean mask for generated tokens
        """
        # Format prompts using chat template
        chat_prompts = []
        for task in tasks:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": task},
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            chat_prompts.append(prompt)
            
        # Tokenize inputs with left padding
        inputs = self.tokenizer(
            chat_prompts,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            return_attention_mask=True,
        ).to(self.model.device)
        
        input_ids = inputs["input_ids"]
        pad_token_id = self.tokenizer.eos_token_id
        
        with torch.no_grad():
            # Configure and run generation
            config = GenerationConfig(do_sample=True, 
                top_p=top_p,
                temperature=temperature, 
                max_length=max_length,
                pad_token_id=pad_token_id,
            )
            sequence_ids = self.model.generate(**inputs, generation_config=config)
            
            # Decode generated sequences
            completions = self.tokenizer.batch_decode(
                sequence_ids[:, input_ids.shape[1]:], skip_special_tokens=True
            )
            
            # Create mask for generated tokens, exclude padding tokens
            action_mask = torch.zeros_like(sequence_ids, dtype=torch.bool)
            action_mask[:, input_ids.shape[1]:] = True
            action_mask[sequence_ids == pad_token_id] = False
            
            return sequence_ids, completions, action_mask[:, 1:]
        
    def compute_log_probs(self, sequence_ids):
        """
        Compute log probabilities for each token in the sequences.
        
        Args:
            sequence_ids (Tensor): Batch of token sequences
            
        Returns:
            Tensor: Log probabilities for each token
        """
        # Create attention mask and position IDs to exclude padding tokens
        attention_mask = sequence_ids != self.tokenizer.pad_token_id
        position_ids = attention_mask.long().cumsum(dim=-1) - 1
        position_ids.masked_fill_(~attention_mask, 0)
        
        # Get model outputs
        output = self.model.forward(input_ids=sequence_ids, 
                                  attention_mask=attention_mask, 
                                  position_ids=position_ids, 
                                  use_cache=False)
        logits = output["logits"][:, :-1]  # Remove last position
        output_ids = sequence_ids[:, 1:]  # Shift right for targets
        
        log_probs = F.log_softmax(logits, dim=-1) 
        selected_log_probs = log_probs.gather(
            dim=-1, 
            index=output_ids.unsqueeze(-1)
        ).squeeze(-1)
        
        return selected_log_probs
