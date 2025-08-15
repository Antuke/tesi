"""This files implements the MoELayerTaskAware module """


import torch, sys, os
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import Callable, Optional
import types 
from dotenv import load_dotenv
load_dotenv()
sys.path.append(os.getenv("REPO_PATH"))
import core.vision_encoder.pe as pe







class TaskAwareGating(nn.Module):
    """Return gating probabilities, given a [batch, num_task, emb_dimension] tensor.
    In our approach, each task has it's associated token """
    def __init__(self, input_dim, num_experts, num_tasks):
        super().__init__()
        self.gating_weights = nn.Parameter(torch.randn(num_tasks, num_experts, input_dim))
    def forward(self, x):
        # [batch, num_task, dimension] @ [num_task, num_experts, dimension] = [batch, num_task, experts]
        return torch.einsum('btd,ted->bte', x, self.gating_weights)

class TaskAgnosticGating(nn.Module):
    def __init__(self, input_dim, num_experts):
        super().__init__()
        self.gating_weights = nn.Parameter(torch.randn(num_experts, input_dim))
    def forward(self, x):
        return torch.einsum('bsd,ed->bse', x, self.gating_weights)
    
class MoELayerTaskAware(nn.Module):
    """Shared Experts, individual gate. Each token received as input is assumed to be a task-embedding produces my the 
    MHCA pooling layer with k-learnable queries, one per task. So each token is routed by an individual gate.
    If task-agnostic is set to true this acts as a normal mixture of expert layer."""
    def __init__(self, input_dim, hidden_dim, output_dim, num_experts, num_tasks, expert_class, top_k=2, task_agnostic_gate=False):
        super().__init__()
        self.num_experts = num_experts
        self.output_dim = output_dim
        self.top_k = top_k
        self.num_tasks = num_tasks
        self.input_dim = input_dim 

        self.experts = nn.ModuleList([expert_class(input_dim, hidden_dim, output_dim) for _ in range(num_experts)])
        self.gating = TaskAwareGating(input_dim, num_experts, num_tasks)
        if task_agnostic_gate:
            self.gating = TaskAgnosticGating(input_dim, num_experts)
            
    def compute_load_balance_loss(self, gating_logits):
        """
        Computes a single, global load balancing loss across all tasks,
        for a shared pool of experts. Globally, expert should be activated uniformaly, but 
        per each task-gate may still prefer only a sub-set of expert without being penalized.
        (e.g Three task gate, the first one prefers Expert 1 & 2, the second one 3 & 4 ecc...)
        """
        # gating_logits has shape (batch_size, num_tasks, num_experts)
        
        # Reshape the logits to treat all tokens from all tasks as one large batch.
        # The new shape is (batch_size * num_tasks, num_experts)
        # [ logits per expert 1, ..., logits per expert 8] -> gating_logits for token1_of_sample1 (token1 is embedding for task1)
        # [ logits per expert 1, ..., logits per expert 8] -> gating_logits for token2_of_sample1 (token2 is embedding for task2)
        # [ logits per expert 1, ..., logits per expert 8] -> gating_logits for token3_of_sample1 (token3 is embedding for task3)
        # [ logits per expert 1, ..., logits per expert 8] -> gating_logits for token1_of_sample2 (token1 is embedding for task1)
        # ...
        gating_logits_flat = gating_logits.reshape(-1, self.num_experts)

        # Calculate routing probabilities for each token
        router_probs = F.softmax(gating_logits_flat, dim=-1)
        
        # Avarage routing probability for each expert
        # [avg_probabily_expert1, avg_probabily_expert2, ..., avg_probabily_expert8] 
        router_avg_prob_per_expert = torch.mean(router_probs, dim=0)
        
        # our goal is to have a global uniform routing
        # so to have avg_probabily_expert1=avg_probabily_expert2= ... =avg_probabily_expert8
        # this does not mean that task-aware gate1 has to have an uniform distribution, but 
        # globally the k task-aware gate should.

        # to achieve a global uniform distribution we use L2 loss. L2 loss is minimum when each element are equals 
        # We multiply by num_experts so to have the minimum loss equal to 1 indipently on the number of experts.
        global_aux_loss = self.num_experts * torch.sum(router_avg_prob_per_expert ** 2)
        
        return global_aux_loss

    def compute_gate_stats(self, top_k_indices):
        """
        Computes the expert activation counts for each gate (task).
        """
        stats = {}

        top_k_indices_cpu = top_k_indices.cpu()

        for i in range(self.num_tasks):
            gate_specific_indices = top_k_indices_cpu[:, i, :].flatten()

            activations = torch.bincount(gate_specific_indices, minlength=self.num_experts)
            
            # The key will be like 'gate_0_activations', 'gate_1_activations', etc.
            stats[f'gate_{i}_activations'] = activations
            
        return stats

    def forward(self, x, calculate_gate_stats=False):
        batch_size, seq_len, _ = x.shape # [batch_size, num_task, embed_dim]
        assert seq_len == self.num_tasks, "Input sequence length must match the number of tasks"

        # task-aware gating
        gating_logits = self.gating(x) # [batch_size, num_task, num_experts]
        aux_loss = 0.0
        gate_stats = {}

        

        # [batch_size, num_task, top_k] 
        top_k_weights, top_k_indices = torch.topk(gating_logits, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_weights, dim=-1)

        if self.training:
            aux_loss = self.compute_load_balance_loss(gating_logits)
        if calculate_gate_stats:
            gate_stats = self.compute_gate_stats(top_k_indices)

        x_flat = x.view(-1, self.input_dim) # [batch_size * num_task, embed_dim]

        # for each token, which expert should it be routed to
        top_k_indices_flat = top_k_indices.view(-1) # [batch_size * num_task, top_k] 

        # Expand the input tensor to align with the top_k routing decisions.
        # Each token is duplicated top_k times.
        expanded_x = x_flat.unsqueeze(1).expand(-1, self.top_k, -1).reshape(-1, self.input_dim)
        # Shape: [batch_size * num_tasks * top_k, embed_dim]
        
        # Route tokens to experts and get outputs
        expert_outputs = torch.zeros(expanded_x.shape, dtype=x.dtype, device=x.device) # [batch_size * num_tasks * top_k, embed_dim]
        for i, expert in enumerate(self.experts):
            # Create a mask for all token assignments for this expert
            mask = (top_k_indices_flat == i)
            # sometimes an expert my not be activated
            if mask.any():
                # Get the expanded tokens assigned to this expert
                tokens_for_expert = expanded_x[mask]
                
                # Calculate the output and place it in the correct rows
                # of our buffer tensor.
                expert_outputs[mask] = expert(tokens_for_expert).to(expert_outputs.dtype)

        # Flatten the weights to align with the expanded tensors
        # [batch_size * num_task * top_k]
        top_k_weights_flat = top_k_weights.view(-1)
        
        # [batch_size * num_tasks * top_k, embed_dim] @ [batch_size * num_task * top_k, 1] 
        weighted_outputs = expert_outputs * top_k_weights_flat.unsqueeze(-1) # unsqueeze for broadcasting
        
        # Reshape to (batch_size * num_tasks, top_k, output_dim) and sum along the top_k dimension
        # This combines the (weighted) contributions from the top_k experts for each original token.
        combined_output_flat = weighted_outputs.view(-1, self.top_k, self.output_dim).sum(dim=1)
        
        # Reshape back to the original batch structure
        final_output = combined_output_flat.view(batch_size, seq_len, self.output_dim)
        
        return final_output, aux_loss, gate_stats

