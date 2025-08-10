"""This files implements the MoELayerTaskAware module"""


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



class ExpertSiglip(nn.Module):
    """Expert network. Same MLP architecture used in Siglip2 (base)"""
    def __init__(self, input_dim=768, hidden_dim=3072, output_dim=768):
        super().__init__()
        self.activation_fn = nn.functional.gelu
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states,approximate="tanh")
        hidden_states = self.fc2(hidden_states)
        return hidden_states

class ExpertPe(nn.Module):
    """Expert network. Same MLP architecture used in Perception Encoders (base)"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ExpertPe, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.gelu = nn.GELU()
    def forward(self, x):
        return self.fc2(self.gelu(self.fc1(x)))


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
    MHCA pooling layer with k-learnable queries, one per task. So each token is routed by an individual gate. """
    def __init__(self, input_dim, hidden_dim, output_dim, num_experts, num_tasks, expert_class, top_k=2, task_agnostic=False):
        super().__init__()
        self.num_experts = num_experts
        self.output_dim = output_dim
        self.top_k = top_k
        self.num_tasks = num_tasks
        self.input_dim = input_dim 

        self.experts = nn.ModuleList([expert_class(input_dim, hidden_dim, output_dim) for _ in range(num_experts)])
        self.gating = TaskAwareGating(input_dim, num_experts, num_tasks)
        if task_agnostic:
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
        expert_outputs = torch.zeros_like(expanded_x) # [batch_size * num_tasks * top_k, embed_dim]
        for i, expert in enumerate(self.experts):
            # Create a mask for all token assignments for this expert
            mask = (top_k_indices_flat == i)
            # sometimes an expert my not be activated
            if mask.any():
                # Get the expanded tokens assigned to this expert
                tokens_for_expert = expanded_x[mask]
                
                # Calculate the output and place it in the correct rows
                # of our buffer tensor.
                expert_outputs[mask] = expert(tokens_for_expert)

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


""" 
 
def convert_attention_pooling_to_moe_siglip(attention_pooling_head, num_tasks: int, num_experts: int, top_k: int, backbone_type: str):
    
    Performs in-place modification of a pre-trained AttentionPooling instance,
    replacing its MLP with a task-aware MoE layer and changing the number of probe queries to the number of task.
    For Perception Encoder.
    
    
    # --- 1. Save the pre-trained MLP weights BEFORE replacing the module ---
    print("Saving pre-trained MLP weights...")
    original_mlp_weights = attention_pooling_head.mlp.state_dict()
    mlp_ratio = attention_pooling_head.mlp.c_fc.out_features // attention_pooling_head.embed_dim

    # --- 2. Resize the probe for the desired number of tasks ---
    print(f"Resizing attention probe from {attention_pooling_head.probe.shape[1]} to {num_tasks} tokens...")
    # We can repeat the original probe to initialize the new ones, preserving knowledge
    original_probe_data = attention_pooling_head.probe.data
    new_probe_data = original_probe_data.repeat(1, num_tasks // original_probe_data.shape[1] + 1, 1)[:, :num_tasks, :]
    attention_pooling_head.probe = nn.Parameter(new_probe_data)

    # --- 3. Build the new MoE layer and seed its experts ---
    expert_class = ExpertPe if backbone_type == 'pe' else ExpertSiglip
    print(f"Creating new MoE layer with {num_experts} experts...")
    moe_layer = MoELayerTaskAware(
        input_dim=attention_pooling_head.embed_dim,
        hidden_dim=int(attention_pooling_head.embed_dim * mlp_ratio),
        output_dim=attention_pooling_head.embed_dim,
        num_experts=num_experts,
        num_tasks=num_tasks,
        top_k=top_k,
        expert_class = expert_class
    )

    # Seed each expert with the pre-trained MLP weights
    for i in range(num_experts):
        moe_layer.experts[i].fc1.weight.data.copy_(original_mlp_weights['c_fc.weight'])
        moe_layer.experts[i].fc1.bias.data.copy_(original_mlp_weights['c_fc.bias'])
        moe_layer.experts[i].fc2.weight.data.copy_(original_mlp_weights['c_proj.weight'])
        moe_layer.experts[i].fc2.bias.data.copy_(original_mlp_weights['c_proj.bias'])
    print(f"All {num_experts} experts have been seeded.")


    attention_pooling_head.mlp = moe_layer


    # The original forward method needs to be replaced to handle the (output, loss) tuple.
    def moe_forward(self, x: torch.Tensor):
        batch, _, _ = x.shape
        q = self.probe.repeat((batch, 1, 1))
        # This part is the same
        attn_output = self.attn(q, x, x, need_weights=False)[0]
        
        # This part changes to handle the tuple
        norm_output = self.layernorm(attn_output)
        moe_output, moe_loss, moe_stats = self.mlp(norm_output) # self.mlp is now the MoE layer
        
        # Residual connection
        final_output = attn_output + moe_output
        
        return final_output, moe_loss, moe_stats 

    # Bind the new forward function to the instance of the model
    attention_pooling_head.forward = types.MethodType(moe_forward, attention_pooling_head)
    #print("Successfully patched the model's forward method.")
    print("--- Model Surgery Complete ---")
    
    return attention_pooling_head



def convert_pe_pooling_to_moe(attention_pooling_head, num_tasks: int, num_experts: int, top_k: int):
    
    Performs in-place modification of a pre-trained AttentionPooling instance,
    replacing its MLP with a task-aware MoE layer and changing the number of probe queries to the number of task.
    For Perception Encoder.
    
    
    # --- 1. Save the pre-trained MLP weights BEFORE replacing the module ---
    print("Saving pre-trained MLP weights...")
    original_mlp_weights = attention_pooling_head.mlp.state_dict()
    mlp_ratio = attention_pooling_head.mlp.c_fc.out_features // attention_pooling_head.embed_dim

    # --- 2. Resize the probe for the desired number of tasks ---
    print(f"Resizing attention probe from {attention_pooling_head.probe.shape[1]} to {num_tasks} tokens...")
    # We can repeat the original probe to initialize the new ones, preserving knowledge
    original_probe_data = attention_pooling_head.probe.data
    new_probe_data = original_probe_data.repeat(1, num_tasks // original_probe_data.shape[1] + 1, 1)[:, :num_tasks, :]
    attention_pooling_head.probe = nn.Parameter(new_probe_data)

    # --- 3. Build the new MoE layer and seed its experts ---
    print(f"Creating new MoE layer with {num_experts} experts...")
    moe_layer = MoELayerTaskAware(
        input_dim=attention_pooling_head.embed_dim,
        hidden_dim=int(attention_pooling_head.embed_dim * mlp_ratio),
        output_dim=attention_pooling_head.embed_dim,
        num_experts=num_experts,
        num_tasks=num_tasks,
        top_k=top_k,
        expert_class = ExpertPe
    )

    # Seed each expert with the pre-trained MLP weights
    for i in range(num_experts):
        moe_layer.experts[i].fc1.weight.data.copy_(original_mlp_weights['fc1.weight'])
        moe_layer.experts[i].fc1.bias.data.copy_(original_mlp_weights['fc1.bias'])
        moe_layer.experts[i].fc2.weight.data.copy_(original_mlp_weights['fc2.weight'])
        moe_layer.experts[i].fc2.bias.data.copy_(original_mlp_weights['fc2.bias'])
    print(f"All {num_experts} experts have been seeded.")


    attention_pooling_head.mlp = moe_layer


    # The original forward method needs to be replaced to handle the (output, loss) tuple.
    def moe_forward(self, x: torch.Tensor):
        batch, _, _ = x.shape
        q = self.probe.repeat((batch, 1, 1))
        # This part is the same
        attn_output = self.attn(q, x, x, need_weights=False)[0]
        
        # This part changes to handle the tuple
        norm_output = self.layernorm(attn_output)
        moe_output, moe_loss, moe_stats = self.mlp(norm_output) # self.mlp is now the MoE layer
        
        # Residual connection
        final_output = attn_output + moe_output
        
        return final_output, moe_loss, moe_stats 

    # Bind the new forward function to the instance of the model
    attention_pooling_head.forward = types.MethodType(moe_forward, attention_pooling_head)
    #print("Successfully patched the model's forward method.")
    print("--- Model Surgery Complete ---")
    
    return attention_pooling_head

from transformers.models.siglip2.modeling_siglip2 import Siglip2MultiheadAttentionPoolingHead

def convert_siglip_pooling_to_moe(
    pooler: Siglip2MultiheadAttentionPoolingHead, 
    num_tasks: int, 
    num_experts: int, 
    top_k: int
):
    
    Performs in-place modification of a pre-trained Siglip2 pooling head,
    replacing its MLP with a task-aware MoE layer.

    Args:
        pooler: An instance of the pre-trained Siglip2MultiheadAttentionPoolingHead.
        num_tasks: The new number of task tokens you want to generate.
        num_experts: The number of experts for the new MoE layer.
        top_k: The number of experts to route to.

    Returns:
        The modified pooler instance.
    
    print("--- Starting Siglip2 Model Surgery ---")
    
    # Extract config from the old MLP
    config = pooler.mlp.config
    hidden_size = config.hidden_size
    intermediate_size = config.intermediate_size

    # 1. Save pre-trained MLP weights
    print("Saving pre-trained SiglipMLP weights...")
    original_mlp_weights = pooler.mlp.state_dict()

    # 2. Resize the probe
    print(f"Resizing attention probe from {pooler.probe.shape[1]} to {num_tasks} tokens...")
    original_probe_data = pooler.probe.data
    # Repeat the original probe to initialize the new ones
    new_probe_data = original_probe_data.repeat(1, num_tasks, 1)
    pooler.probe = nn.Parameter(new_probe_data)

    # 3. Build the new MoE layer and seed its experts
    print(f"Creating new MoE layer with {num_experts} experts...")

    moe_layer = MoELayerTaskAware(
        input_dim=hidden_size,
        hidden_dim=intermediate_size,
        output_dim=hidden_size,
        num_experts=num_experts,
        num_tasks=num_tasks,
        top_k=top_k,
        expert_class = ExpertSiglip
    )

    # Seed each expert with the pre-trained SiglipMLP weights
    for i in range(num_experts):
        moe_layer.experts[i].fc1.weight.data.copy_(original_mlp_weights['fc1.weight'])
        moe_layer.experts[i].fc1.bias.data.copy_(original_mlp_weights['fc1.bias'])
        moe_layer.experts[i].fc2.weight.data.copy_(original_mlp_weights['fc2.weight'])
        moe_layer.experts[i].fc2.bias.data.copy_(original_mlp_weights['fc2.bias'])
    print(f"All {num_experts} experts have been seeded.")

    # 4. Replace the old MLP with the new MoE layer
    pooler.mlp = moe_layer
    print("Replaced original SiglipMLP with the new MoE layer.")

    # 5. Monkey-patch the forward method
    # Copied from the original source to ensure attention_mask logic is preserved
    def moe_forward(self, hidden_state: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        batch_size = hidden_state.shape[0]
        # The resized probe will now produce num_tasks tokens
        probe = self.probe.repeat(batch_size, 1, 1)

        # The original attention mask logic should be preserved as-is
        if attention_mask is not None:
            # This part of the code is specific to how Siglip handles masks
            # and should not be changed.
            target_len, source_len = probe.shape[1], hidden_state.shape[1]
            # This helper function is likely defined within the transformers library
            # If not, you may need to copy it. For now, assume it's accessible.
            from transformers.models.siglip.modeling_siglip import _prepare_4d_attention_mask
            attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_state.dtype, target_len)
            attention_mask = attention_mask.repeat(1, self.num_heads, target_len, 1)
            attention_mask = attention_mask.reshape(-1, target_len, source_len)

        # The output of attention now has shape (batch_size, num_tasks, hidden_size)
        attn_output = self.attention(probe, hidden_state, hidden_state, attn_mask=attention_mask)[0]

        # --- THIS IS THE MODIFIED PART ---
        residual = attn_output
        layernorm_output = self.layernorm(attn_output)
        
        # Unpack the tuple from our new MoE layer
        moe_output, moe_loss, moe_stats = self.mlp(layernorm_output)
        
        hidden_state = residual + moe_output
        
        # CRUCIAL CHANGE: Return all task tokens, not just the first one.
        # And also return the loss and stats.
        return (hidden_state, moe_loss, moe_stats)

    # Bind the new forward function to the instance of the pooler
    pooler.forward = types.MethodType(moe_forward, pooler)
    print("Successfully patched the pooler's forward method.")
    print("--- Siglip2 Model Surgery Complete ---")
    
    return pooler


"""