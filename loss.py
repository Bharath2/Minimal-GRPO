import torch
import torch.nn as nn

class PolicyLoss(nn.Module):
    """
    This class implements a loss function that combines PPO-style policy optimization with 
    KL-divergence regularization from a reference model. It is designed for training language
    models with reward signals. Based on Group Relative Policy Optimization (GRPO) from DeepSeekMath.

    Args:
        clip_eps (float, optional): Clipping parameter for PPO loss. Defaults to 0.2.
        kl_weight (float, optional): Weight for KL divergence term. Defaults to 0.05.
    """

    def __init__(self, clip_eps = 0.2, kl_weight = 0.05,  max_len = None):
        super().__init__()
        self.clip_eps = clip_eps
        self.kl_weight = kl_weight
        self.max_len = max(1, max_len) if max_len is not None else None
        
    def forward(self, log_probs, old_log_probs, advantages, ref_log_probs = None, action_mask = None): 
        """Forward pass to compute the Policy Loss.

        Args:
            log_probs (Tensor): Log probabilities from current policy, shape [B, seq_len]
            old_log_probs (Tensor): Log probabilities from old policy, shape [B, seq_len]
            advantages (Tensor): Advantage estimates, shape [B]
            ref_log_probs (Tensor, optional): Log probs from reference model for KL term
            action_mask (Tensor, optional): Mask for valid actions, same shape as log_probs

        Returns:
            Tensor: Scalar loss value combining PPO and KL objectives
        """
        if advantages.dim() == 1:  # [B] -> [B, 1]
            advantages = advantages.unsqueeze(1) 

        # PPO loss: Compute the probability ratio and surrogate objectives
        ratio = (log_probs - old_log_probs).exp()
        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advantages
        loss = -torch.min(surr1, surr2)
        
        # Compute the KL divergence from reference model using f(u) = u - log(u) - 1
        if ref_log_probs is not None:
            log_ratio = ref_log_probs - log_probs
            kl = log_ratio.exp() - log_ratio - 1
            loss = loss + self.kl_weight * kl

        # If action_mask is not None, use masked average, otherwise use mean
        action_length = log_probs.shape[1]
        if action_mask is not None:
            loss = loss * action_mask
            action_length = action_mask.sum(axis = 1)
        # if max_len is specified, use max_len to normalize (DR.GRPO)
        if self.max_len is not None: action_length = self.max_len
        avg_loss = (loss.sum(axis = 1) / (action_length + 1e-3)).mean()
        return avg_loss
