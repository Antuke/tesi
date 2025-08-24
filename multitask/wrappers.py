"""This file contains the wrapper classes to adapt the Vision Transformers of SigLip2 and Percepetion Encoder
to achieve k-learnable query probing for multi-task training and substitution of the last dense layer with a 
task-aware Mixture of expert layer. To retain the pre-trained information all the new parameters added do not start with 
random weights, but copy the weights of the original layer that they substitute (so the k-probe start with the weight of each the single typical probe used 
and the MoE MLP start with the same weight of the original dense layer)"""


import torch
from multitask.moe_task_aware import *
from transformers.models.siglip2.modular_siglip2 import Siglip2MultiheadAttentionPoolingHead
from dotenv import load_dotenv
load_dotenv()
REPO_PATH = os.getenv("REPO_PATH")
if REPO_PATH:
    sys.path.append(REPO_PATH)

from core.vision_encoder import pe




# ------------- SIGLIP2 ------------- #
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


class SigLIPKProbeHead(nn.Module):
    """
    This module wraps the original SigLIP head layers to return all probe outputs.
    """
    def __init__(self, original_head: Siglip2MultiheadAttentionPoolingHead):
        super().__init__()
        print("[WRAPPER] Substituing the Attention pooling head of SigLip2 with k-probes")
        # Copy necessary layers and parameters from the original head
        self.probe = original_head.probe
        self.attention = original_head.attention
        
        self.layernorm = original_head.layernorm
        self.mlp = original_head.mlp
        # self.proj = nn.Parameter(torch.randn(3, 768, 1024),requires_grad=True)

    def forward(self, hidden_state: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = hidden_state.shape[0]
        # The k-probes are already part of self.probe, so we just repeat for the batch
        probe = self.probe.repeat(batch_size, 1, 1)


        if attention_mask is not None:
            raise ValueError("Custom SigLIP k-probe head does not support attention_mask.")

        hidden_state, attn_weights = self.attention(probe, hidden_state, hidden_state)
        residual = hidden_state
        hidden_state = self.layernorm(hidden_state)
        hidden_state = residual + self.mlp(hidden_state)
        
        # hidden_state = torch.einsum('btd,tdh->bth', hidden_state, self.proj)
        # Return the entire tensor of shape [batch_size, num_probes, hidden_dim]
        
        return hidden_state, attn_weights

class SigLIPKMoeHead(nn.Module):
    """
    A wrapper class that converts a standard Siglip2 pooling head into a
    multi-task MoE-based head.
    """
    def __init__(
        self, 
        original_pooler: Siglip2MultiheadAttentionPoolingHead, 
        num_tasks: int,
        num_experts: int,
        top_k: int,
        task_agnostic_gate: bool
    ):
        super().__init__()
        print("[WRAPPER] Substituing the Attention pooling head of SigLip2 with MOE and k-probes")

        self.attention = original_pooler.attention
        # self.num_heads = original_pooler.num_heads 
        self.layernorm = original_pooler.layernorm
    

        original_probe_data = original_pooler.probe.data
        new_probe_data = original_probe_data.repeat(1, num_tasks, 1)
        self.probe = nn.Parameter(new_probe_data)

        
        config = original_pooler.mlp.config
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size
        

        # new MoE layer
        self.mlp = MoELayerTaskAware(
            input_dim=hidden_size,
            hidden_dim=intermediate_size,
            output_dim=hidden_size,
            num_experts=num_experts,
            num_tasks=num_tasks,
            top_k=top_k,
            expert_class = ExpertSiglip,
            task_agnostic_gate = task_agnostic_gate
        )
        
        # Seed the new experts with the weights from the original MLP
        original_mlp_weights = original_pooler.mlp.state_dict()
        for i in range(num_experts):
            self.mlp.experts[i].fc1.weight.data.copy_(original_mlp_weights['fc1.weight'])
            self.mlp.experts[i].fc1.bias.data.copy_(original_mlp_weights['fc1.bias'])
            self.mlp.experts[i].fc2.weight.data.copy_(original_mlp_weights['fc2.weight'])
            self.mlp.experts[i].fc2.bias.data.copy_(original_mlp_weights['fc2.bias'])

        # To match PE method, we add a projection to 1024
        # self.proj = nn.Parameter(torch.randn(3, 768, 1024),requires_grad=True)

    def forward(self, hidden_state: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, calculate_gate_stats = True):
        """
        The new forward pass that handles MoE logic.
        """
        batch_size = hidden_state.shape[0]
        probe = self.probe.repeat(batch_size, 1, 1)

        if attention_mask is not None:
            raise ValueError("Custom SigLIP k-probe head does not support attention_mask.")

        attn_output, attn_weights = self.attention(probe, hidden_state, hidden_state, attn_mask=attention_mask)

        residual = attn_output
        layernorm_output = self.layernorm(attn_output)
        
        moe_output, moe_loss, moe_stats = self.mlp(layernorm_output, calculate_gate_stats)
        
        hidden_state = residual + moe_output

        # to match pe
        # hidden_state = torch.einsum('btd,tdh->bth', hidden_state, self.proj)

 

        return hidden_state, moe_loss, moe_stats, attn_weights










# ------------- PERCEPTION ENCODERS ------------- #
class ExpertPe(nn.Module):
    """Expert network. Same MLP architecture used in Perception Encoders (base)"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ExpertPe, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.gelu = nn.GELU()
    def forward(self, x):
        return self.fc2(self.gelu(self.fc1(x)))
        

class PEMoeViT(nn.Module):
    """
    A top-level wrapper that takes a pre-trained VisionTransformer and substitutes
    its attention pooling submodule to the MoE-based version.
    """
    def __init__(
        self,
        original_vit: pe.VisionTransformer, # Expects a VisionTransformer instance
        num_tasks: int,
        num_experts: int,
        top_k: int,
        task_agnostic_gate: bool
    ):
        super().__init__()
        print("[WRAPPER] Initializing MoEVisionTransformer Wrapper")

        if not hasattr(original_vit, 'pool_type') or original_vit.pool_type != "attn":
            raise ValueError("MoE conversion only supports VisionTransformer with pool_type='attn'.")


        self.conv1 = original_vit.conv1
        self.transformer = original_vit.transformer
        self.ln_pre = original_vit.ln_pre
        self.ln_post = original_vit.ln_post
        self.pool_type = original_vit.pool_type
        self.use_cls_token = original_vit.use_cls_token
        self.use_abs_posemb = original_vit.use_abs_posemb
        self.patch_size = original_vit.patch_size
        self.posemb_grid_size = original_vit.posemb_grid_size if hasattr(original_vit, 'posemb_grid_size') else None
        self.positional_embedding = original_vit.positional_embedding if hasattr(original_vit, 'positional_embedding') else None
        self.class_embedding = original_vit.class_embedding if hasattr(original_vit, 'class_embedding') else None
        self.rope = original_vit.rope
        self.width = original_vit.width
        self.proj_dim = original_vit.proj_dim

        print("[WRAPPER] Substituing the original AttentionPooling layer to MoEAttentionPooling...")
        self.attn_pool = MoEAttentionPooling(
            original_pooler=original_vit.attn_pool,
            num_tasks=num_tasks,
            num_experts=num_experts,
            top_k=top_k,
            task_agnostic_gate=task_agnostic_gate
        ).to('cuda')

        # three distinct projection matrices
        proj_tensor_replicated = original_vit.proj.unsqueeze(0).repeat(num_tasks, 1, 1)
        proj_tensor_replicated = proj_tensor_replicated.clone().detach().to('cuda')
        self.proj = nn.Parameter(proj_tensor_replicated, requires_grad=True)
        print("[WRAPPER] Wrapper Initialization Complete")
    
    # copied from original ViT of pe
    def _sample_abs_posemb(self, grid_h: int, grid_w: int):
        """Interpolates the absolute position embedding if necessary."""
        if self.posemb_grid_size == grid_h and self.posemb_grid_size == grid_w:
            return self.positional_embedding[None, ...]

        pos_embed = self.positional_embedding
        if self.use_cls_token:
            cls_token_embed, pos_embed = pos_embed[:1], pos_embed[1:]

        pos_embed = (
            pos_embed.reshape(1, self.posemb_grid_size, self.posemb_grid_size, -1)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        pos_embed = F.interpolate(
            pos_embed, size=(grid_h, grid_w), mode="bilinear", align_corners=False
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(-1, self.width).contiguous()

        if self.use_cls_token:
            pos_embed = torch.cat([cls_token_embed, pos_embed], dim=0)

        return pos_embed[None, ...]

    # copied from original ViT of pe
    def forward_features(self, x: torch.Tensor, **kwargs):
        batch, _, h, w = x.shape
        grid_h, grid_w = h // self.patch_size, w // self.patch_size
        x = self.conv1(x).permute(0, 2, 3, 1).reshape(batch, -1, self.width)
        if self.use_cls_token:
            x = torch.cat([self.class_embedding.view(1, 1, -1).expand(batch, -1, -1), x], dim=1)
        if self.use_abs_posemb:
            x = x + self._sample_abs_posemb(grid_h, grid_w)
        if self.rope:
            self.rope.update_grid(x.device, grid_h, grid_w)
        x = self.ln_pre(x)
        x = self.transformer(x, **kwargs)
        x = self.ln_post(x)
        return x

    def _pool(self, x: torch.Tensor, return_attn_weights:bool = False):
        """The pooling logic, now using our MoE pooler."""
        return self.attn_pool(x, return_attn_weights)

    def forward(self, x: torch.Tensor, return_attn_weights:bool =False, **kwargs):
        """The main forward pass for the wrapped MoE Vision Transformer."""
        x = self.forward_features(x, **kwargs)
        
        # Pooling step now returns three values
        x, loss, gate_stats,attn_weights = self._pool(x, return_attn_weights=return_attn_weights)
        
        # The projection layer is applied only to the final tensor output
        if self.proj is not None:
            # Einsum handles the projection on the last dimension of the multi-token output
            x = torch.einsum('btd,tdh->bth', x, self.proj)
        
        print(x.shape)
        return x, loss, gate_stats, attn_weights


class MoEAttentionPooling(nn.Module):
    """
    A wrapper that replaces the MLP in a standard AttentionPooling layer
    with a multi-task MoE layer. Also implements k-probing
    """
    def __init__(
        self,
        original_pooler: nn.Module,
        num_tasks: int,
        num_experts: int,
        top_k: int,
        task_agnostic_gate: bool
    ):
        super().__init__()
        self.attention = original_pooler.attn
        self.layernorm = original_pooler.layernorm
        original_probe_data = original_pooler.probe.data

        new_probe_data = original_probe_data.repeat(1, num_tasks // original_probe_data.shape[1] + 1, 1)[:, :num_tasks, :]
        self.probe = nn.Parameter(new_probe_data)

        #  Build and seed the MoE-based MLP replacement
        embed_dim = original_pooler.embed_dim
        self.mlp = MoELayerTaskAware(
            input_dim=embed_dim,
            hidden_dim=(embed_dim * 4),
            output_dim=embed_dim,
            num_experts=num_experts,
            num_tasks=num_tasks,
            top_k=top_k,
            expert_class = ExpertPe,
            task_agnostic_gate = task_agnostic_gate
        )
        
        original_mlp_weights = original_pooler.mlp.state_dict()
        for i in range(num_experts):
            self.mlp.experts[i].fc1.weight.data.copy_(original_mlp_weights['c_fc.weight'])
            self.mlp.experts[i].fc1.bias.data.copy_(original_mlp_weights['c_fc.bias'])
            self.mlp.experts[i].fc2.weight.data.copy_(original_mlp_weights['c_proj.weight'])
            self.mlp.experts[i].fc2.bias.data.copy_(original_mlp_weights['c_proj.bias'])

    def forward(self, x: torch.Tensor,return_attn_weights: bool = False) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """The new forward pass that handles MoE logic."""
        batch_size = x.shape[0]
        probe = self.probe.repeat(batch_size, 1, 1)
        
        attn_output, attn_weights = self.attention(probe, x, x, need_weights=return_attn_weights)
        residual = attn_output
        layernorm_output = self.layernorm(attn_output)
        
        # self.mlp is now our MoE layer, which returns a 3-part tuple
        moe_output, moe_loss, moe_stats = self.mlp(layernorm_output)
        
        final_output = residual + moe_output

        return final_output, moe_loss, moe_stats, attn_weights