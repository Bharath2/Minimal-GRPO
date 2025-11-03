import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
)
from peft import get_peft_model, LoraConfig, TaskType

try:
    import flash_attn
    FLASH_ATTN_AVAILABLE = True
except (ImportError, Exception) as e:
    FLASH_ATTN_AVAILABLE = False
    print(f"Note: Flash Attention not available")


class HuggingFaceLM:
    """
    A wrapper class for Hugging Face causal language models that handles text generation and log probability computation.
    
    Args:
        model_identifier (str): Path or identifier for the pretrained model (e.g., "meta-llama/Llama-3.2-1B-Instruct").
        bf16 (bool): Whether to use bfloat16 precision; defaults to True.
        device (str or torch.device): Device to load the model on (e.g., "cuda" or torch.device("cuda")).
        system_prompt (str, optional): A system prompt to prepend to all inputs.
        model_class (callable, optional): The model class to use; defaults to AutoModelForCausalLM.
        use_lora (bool): Whether to use LoRA for parameter-efficient fine-tuning; defaults to False.
        lora_config (dict, optional): Configuration for LoRA (required if use_lora=True).
        cache_dir (str, optional): Directory to cache downloaded models. If None, uses default HF cache.
    """
    def __init__(
        self,
        model_identifier="meta-llama/Llama-3.2-1B-Instruct",
        bf16=True,
        device=torch.device("cuda"),
        system_prompt=None,
        model_class=None,
        use_lora=False,
        lora_config=None,
        cache_dir=None
    ):
        # Check if CUDA is available when GPU is requested
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            print("WARNING: CUDA device requested but CUDA is not available. Falling back to CPU.")
            device = torch.device("cpu")
        
        # Initialize tokenizer with padding token
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_identifier,
            cache_dir=cache_dir
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        if model_class is None:
            model_class = AutoModelForCausalLM
        
        # Load model with specified configuration
        model_kwargs = {
            "trust_remote_code": False,
            "dtype": torch.bfloat16 if bf16 else "auto",
            "device_map": device,
            "cache_dir": cache_dir,
        }
        
        # Use Flash Attention 2 for CUDA devices if available (optional but recommended)
        if torch.cuda.is_available() and FLASH_ATTN_AVAILABLE:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        self.model = model_class.from_pretrained(
            model_identifier,
            **model_kwargs
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
            # Enable gradient checkpointing for PEFT models
            self.model.enable_input_require_grads()

        self.model.config.use_cache = False

        
    def generate(self, tasks, temperature=1.0, top_p=0.75, max_length=640, max_new_tokens=None, do_sample=True):
        """
        Generate completions for a batch of tasks.
        
        Args:
            tasks (List[str]): List of input prompts
            temperature (float): Sampling temperature. Defaults to 1.0.
            top_p (float): Nucleus sampling - consider top tokens with cumulative probability mass of top_p. Defaults to 0.75.
            max_length (int): Maximum total sequence length (input + output). Defaults to 640.
            max_new_tokens (int): Maximum number of new tokens to generate (output only). If specified, overrides max_length.
            do_sample (bool): Whether to use sampling or greedy decoding. If False, uses greedy decoding. Defaults to True.
            
        Returns:
            tuple: (sequence_ids, completions, action_mask) containing:
                - sequence_ids (Tensor): Token IDs for full sequences
                - completions (List[str]): Generated text completions
                - action_mask (Tensor): Boolean mask for generated tokens
        """
        # Format prompts using chat template (optimized with list comprehension)
        chat_prompts = [
            self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": task},
                ],
                tokenize=False,
                add_generation_prompt=True
            )
            for task in tasks
        ]
            
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
            # Configure and run generation (optimized config creation)
            config_params = {
                'do_sample': do_sample,
                'pad_token_id': pad_token_id,
            }
            # Only add temperature and top_p if sampling is enabled
            if do_sample:
                config_params['top_p'] = top_p
                config_params['temperature'] = temperature
            
            # Use max_new_tokens if specified, otherwise fall back to max_length
            if max_new_tokens is not None:
                config_params['max_new_tokens'] = max_new_tokens
            else:
                config_params['max_length'] = max_length
            
            config = GenerationConfig(**config_params)
            sequence_ids = self.model.generate(**inputs, generation_config=config)
            
            # Decode generated sequences
            completions = self.tokenizer.batch_decode(
                sequence_ids[:, input_ids.shape[1]:], skip_special_tokens=True
            )
            
            # Create mask for generated tokens, exclude padding tokens (optimized)
            input_len = input_ids.shape[1]
            action_mask = (sequence_ids != pad_token_id)  # Mask out padding
            action_mask[:, :input_len] = False  # Mask out input tokens
            
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
        
        # Get model outputs with minimal memory footprint
        output = self.model.forward(
            input_ids=sequence_ids, 
            attention_mask=attention_mask, 
            position_ids=position_ids, 
            use_cache=False,
            return_dict=True
        )
        
        # Extract logits and clear output dict to free memory
        logits = output.logits[:, :-1].contiguous()
        del output
        
        # Shift sequence for targets
        output_ids = sequence_ids[:, 1:].contiguous()
        
        # Compute log probs in-place where possible
        log_probs = F.log_softmax(logits, dim=-1)
        del logits  # Free memory immediately
        
        # Gather selected log probs efficiently
        selected_log_probs = log_probs.gather(
            dim=-1, 
            index=output_ids.unsqueeze(-1)
        ).squeeze(-1)
        del log_probs  # Free memory immediately
        
        return selected_log_probs

