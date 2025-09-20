
from lora import MTLoRALinear
from einops import rearrange
from torch.nn import functional as F
from dotenv import load_dotenv
import os, sys

load_dotenv()
REPO_PATH = os.getenv("REPO_PATH")
if REPO_PATH:
    sys.path.append(REPO_PATH)

from core.vision_encoder.pe import SelfAttention, ResidualAttentionBlock, AttentionPooling   
from lora import MTLoRALinear, MTLoRAQKV
from einops import rearrange
from torch.nn import functional as F
import torch.nn as nn
from typing import Any,  Dict, List, Tuple, Type
from config.task_config import Task
import torch
from typing import Optional, Dict, Tuple, Union, Mapping,OrderedDict
from multitask.lora import *

DROPOUT_P = 0.5
class MTLModel(nn.Module):
    def __init__(self, backbone, tasks: List[Task], rank: int = 16, use_lora: bool = True):
        super().__init__()
        self.tasks = tasks
        
        # log_vars is for uncertainty weighting
        self.log_vars =  nn.Parameter(torch.zeros(len(tasks)))
        task_names = [task.name for task in tasks]

        # save last residual attention block, as we need the weights values to seed the new mtl version
        orig_last_block = backbone.transformer.resblocks[-1]
        self.ln_post = backbone.ln_post

        # save the attention pooling, as we need the weights values to seed the task specifics attention pooling layers
        orig_attn_pool = backbone.attn_pool

        
        width = backbone.width
        heads = backbone.heads
        rope = backbone.rope
        
        
        self.backbone = backbone
        self.backbone.truncate(layer_idx=22) # 23th block becomes the last (the idx is 22)

        # mtl block that produces t-task specific features maps, plus a shared one
        self.mtl_layer = MTLoRAResidualAttentionBlock(
            d_model=width,
            n_head=heads,
            rope=rope,
            r={'shared': rank, **{name: rank for name in task_names}},
            tasks=task_names,
            shared_mode='matrix' 
        )
        

        self.mtl_layer.load_from_original_block(orig_last_block)
        print("MTL-LoRA final block created and initialized from pretrained weights.")

        
        self.task_specific_attn_pool = nn.ModuleDict({
            task.name: AttentionPooling(embed_dim=width, num_heads=8)
            for task in self.tasks
        })
        

        for task in self.tasks:
            self.task_specific_attn_pool[task.name].load_state_dict(orig_attn_pool.state_dict())
        print("Task-specific Attention Pooling layers created and initialized.")

        
        self.prediction_layers = nn.ModuleDict({
            task.name: nn.Sequential(
                nn.BatchNorm1d(width),
                nn.Dropout(p=DROPOUT_P),  
                nn.Linear(width, len(task.class_labels))
            )
            for task in self.tasks
        })

        self.backbone.del_muda()
        del self.backbone.attn_pool


        if use_lora:
            add_lora_to_backbone(self.backbone, rank=rank, lora_alpha=8)

        
        print("Task-specific prediction heads created.")


    def enable_gradient_checkpointing(self):
        """Call this method after setting up parameter requires_grad"""
        backbone_has_trainable = any(param.requires_grad for param in self.backbone.parameters())
        if backbone_has_trainable:
            self.backbone.set_grad_checkpointing()
            print("Gradient checkpointing enabled for backbone (has trainable parameters)")
        else:
            print("Gradient checkpointing not enabled - backbone has no trainable parameters")

    def forward(self, x: torch.Tensor):
        # Shared feature map from the backbone
        # norm=False, because normalization is "trained" on the feature map of the output of the last ResidualAttentionBlock
        # so we will normalize the task specific feature map, instead of the shared one
        # strip_cls_token=False, because in the PE paper it has been shown to be beneficial to keep it
        features = self.backbone.forward_features(x, norm=False, strip_cls_token=False) 

        # Equal for each task, as our mtl layer follows a task-agnostic layer
        task_features_input = {task.name: features for task in self.tasks}

        # Returns also a shared features map, that is discarded, 
        # task features is a dictionary, the key is task name, and the value is a tensor of shape (batch_size, n_tokens, d_model)
        # rappresting the task specific features map
        _, task_features  = self.mtl_layer(features, x_tasks=task_features_input)


        logits = {}
        for task in self.tasks:
            task_name = task.name
            
            # Layer norm from pre-trained ViT
            feat = self.ln_post(task_features[task_name])
            
            # Attention pooling to obtain a task-specific embedding to classify
            pooled_feat = self.task_specific_attn_pool[task_name](feat)
            pooled_feat = pooled_feat.squeeze(1) # (batch, 1, d_model) -> (batch, d_model)
            
            logits[task_name] = self.prediction_layers[task_name](pooled_feat)
            
        return logits

    def save_model(self, filepath: str):
        print(f"Saving model state_dict to {filepath}")
        torch.save(self.state_dict(), filepath)

    def load_model(self, filepath: str):
        print(f"Loading model state_dict from {filepath}")
        self.load_state_dict(torch.load(filepath))

class MTLoRAResidualAttentionBlock(nn.Module):
    """Adaptation of Perception Encoder ResidualAttentionBlock with MTLora, to produce t-task specific feature-maps and a shared feature map"""
    def __init__(
        self,
        d_model: int,
        n_head: int,
        mlp_ratio: float = 4.0,
        ls_init_value: float = None,
        act_layer =  nn.GELU,
        norm_layer = nn.LayerNorm,
        drop_path: float = 0.0,
        rope: Optional[nn.Module] = None, 
        r: Union[int, Mapping[str, int]] = 0,
        lora_shared_scale: float = 1.0,
        lora_task_scale: float = 1.0,
        lora_dropout: float = DROPOUT_P,
        tasks=None,
        trainable_scale_shared=False,
        trainable_scale_per_task=True,
        shared_mode: str = 'matrix',
    ):
        super().__init__()
        self.tasks = tasks
        self.num_heads = n_head
        self.head_dim = d_model // n_head
        self.scale = self.head_dim ** -0.5
        self.rope = rope 

        # MultiTask Lora for QKV matrices
        # (MTLoRAQKV does not actually compute attention, but returns the shared QKV matrices and the task-specific QKV matrices)
        self.attn = MTLoRAQKV(
            in_features=d_model,
            out_features=d_model,
            r=r, lora_shared_scale=lora_shared_scale, lora_task_scale=lora_task_scale,
            lora_dropout=lora_dropout, tasks=tasks, trainable_scale_shared=trainable_scale_shared,
            trainable_scale_per_task=trainable_scale_per_task, shared_mode=shared_mode
        )

        # MultiTask Lora for projection matrices in mha
        self.out_proj = MTLoRALinear(
            in_features=d_model,
            out_features=d_model,
            r=r, lora_shared_scale=lora_shared_scale, lora_task_scale=lora_task_scale,
            lora_dropout=lora_dropout, tasks=tasks, trainable_scale_shared=trainable_scale_shared,
            trainable_scale_per_task=trainable_scale_per_task, shared_mode=shared_mode
        )

        self.ls_1 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()
        self.ls_2 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

        self.ln_1 = norm_layer(d_model)
        self.ln_2 = norm_layer(d_model)

        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # LoRA-enabled MLP
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            OrderedDict([
                ("c_fc", MTLoRALinear(
                    d_model, mlp_width, r=r, lora_shared_scale=lora_shared_scale,
                    lora_task_scale=lora_task_scale, lora_dropout=lora_dropout, tasks=tasks,
                    trainable_scale_shared=trainable_scale_shared, trainable_scale_per_task=trainable_scale_per_task,
                    shared_mode=shared_mode
                )),
                ("gelu", act_layer()),
                ("c_proj", MTLoRALinear(
                    mlp_width, d_model, r=r, lora_shared_scale=lora_shared_scale,
                    lora_task_scale=lora_task_scale, lora_dropout=lora_dropout, tasks=tasks,
                    trainable_scale_shared=trainable_scale_shared, trainable_scale_per_task=trainable_scale_per_task,
                    shared_mode=shared_mode
                )),
            ])
        )

    def _call_attn(
        self,
        x_shared: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        x_tasks: Optional[Dict[str, torch.Tensor]] = None,
    ):
        # s is the number of patches/tokens, sequence length
        proj, proj_tasks = self.attn(x_shared, x_tasks) # proj is (b s 3*d_model), proj_tasks is dict of (b s 3*d_model), one entry per task

        def compute_attention(projection_tensor):
            # Reshape Q, K, V
            # projection_tensor is (b s 3*d_model), need to split and rearrange
            _, s, _ = projection_tensor.shape
            # output_features from MTLoRAQKV is d_model, so 3 * d_model
            split_size = self.attn.q.linear.out_features # This should be d_model
            
            # Unflatten into (b s 3 d_model) then transpose to get (3 b s d_model)
            q, k, v = projection_tensor.unflatten(-1, (3, split_size)).permute(2, 0, 1, 3).contiguous()
            # Rearrange for multi-head attention (b h s d)
            q = rearrange(q, "b s (h d) -> b h s d", h=self.num_heads)
            k = rearrange(k, "b s (h d) -> b h s d", h=self.num_heads)
            v = rearrange(v, "b s (h d) -> b h s d", h=self.num_heads)

            if self.rope: 
                q, k = self.rope(q, k)

            attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, scale=self.scale)
            return rearrange(attn_output, "b h s d -> b s (h d)")

        # Process shared path
        attn_result = compute_attention(proj)

        # Process task-specific paths
        attn_tasks_results = {}
        if proj_tasks:
            for task, task_proj in proj_tasks.items():
                attn_tasks_results[task] = compute_attention(task_proj)

        # Apply output projection
        # out_proj is an MTLoRALinear, so its forward expects (x, x_tasks)
        shared_out, tasks_out = self.out_proj(attn_result, x_tasks=attn_tasks_results if attn_tasks_results else None)

        return shared_out, tasks_out

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        x_tasks: Optional[Dict[str, torch.Tensor]] = None,
    ):
        # Attention block
        norm_x = self.ln_1(x)
        norm_x_tasks = {task: self.ln_1(x_tasks[task]) for task in self.tasks} if x_tasks else None

        attn_out, attn_tasks_out = self._call_attn(norm_x, attn_mask=attn_mask, x_tasks=norm_x_tasks)

        x = x + self.drop_path1(self.ls_1(attn_out))
        if attn_tasks_out and x_tasks:
            for task in self.tasks:
                x_tasks[task] = x_tasks[task] + self.drop_path1(self.ls_1(attn_tasks_out[task]))

        # MLP block
        norm_x = self.ln_2(x)
        norm_x_tasks = {task: self.ln_2(x_tasks[task]) for task in self.tasks} if x_tasks else None

        # The MTLoRALinear forward needs to be called directly for the sequential MLP
        mlp_fc_out, mlp_fc_tasks_out = self.mlp.c_fc(norm_x, norm_x_tasks)
        gelu_out = self.mlp.gelu(mlp_fc_out)
        gelu_tasks_out = {task: self.mlp.gelu(mlp_fc_tasks_out[task]) for task in self.tasks} if mlp_fc_tasks_out else None

        mlp_proj_out, mlp_proj_tasks_out = self.mlp.c_proj(gelu_out, gelu_tasks_out)

        x = x + self.drop_path2(self.ls_2(mlp_proj_out))
        if mlp_proj_tasks_out and x_tasks:
            for task in self.tasks:
                x_tasks[task] = x_tasks[task] + self.drop_path2(self.ls_2(mlp_proj_tasks_out[task]))

        return x, x_tasks

    def load_from_original_block(self, original_block):
        """
        Initializes the weights of this block from a pre-trained ResidualAttentionBlock.
        The LoRA-specific parameters are reset to their initial state (delta = 0).
        """
        with torch.no_grad():
            # Copy LayerNorm and LayerScale weights
            self.ln_1.load_state_dict(original_block.ln_1.state_dict())
            self.ln_2.load_state_dict(original_block.ln_2.state_dict())
            self.ls_1.load_state_dict(original_block.ls_1.state_dict())
            self.ls_2.load_state_dict(original_block.ls_2.state_dict())

            # Copy MLP weights into the .linear attribute of the MTLoRALinear layers
            self.mlp.c_fc.linear.load_state_dict(original_block.mlp.c_fc.state_dict())
            self.mlp.c_proj.linear.load_state_dict(original_block.mlp.c_proj.state_dict())

            # Copy Attention weights
            # Both SelfAttention and nn.MultiheadAttention store QKV weights combined
            if isinstance(original_block.attn, SelfAttention):
                # Using migrate_weights ensures the Parameters are copied to the Linear layer first
                # Then we can extract from the Linear layer
                original_block.attn.migrate_weights() # Ensure weights are in .in_proj and .out_proj
                
                # Split the combined weight and bias tensors into Q, K, V from .in_proj
                qkv_weight = original_block.attn.in_proj.weight
                qkv_bias = original_block.attn.in_proj.bias

                q_w, k_w, v_w = qkv_weight.chunk(3)
                q_b, k_b, v_b = qkv_bias.chunk(3)

                # Load into the .linear attributes of the MTLoRAQKV module
                self.attn.q.linear.weight.copy_(q_w)
                self.attn.q.linear.bias.copy_(q_b)

                self.attn.k.linear.weight.copy_(k_w)
                self.attn.k.linear.bias.copy_(k_b)

                self.attn.v.linear.weight.copy_(v_w)
                self.attn.v.linear.bias.copy_(v_b)

                # Load the output projection weights
                self.out_proj.linear.load_state_dict(original_block.attn.out_proj.state_dict())
            elif isinstance(original_block.attn, nn.MultiheadAttention):
                self.attn.q.linear.weight.copy_(original_block.attn.in_proj_weight[:self.attn.q.linear.out_features, :])
                self.attn.q.linear.bias.copy_(original_block.attn.in_proj_bias[:self.attn.q.linear.out_features])
                
                self.attn.k.linear.weight.copy_(original_block.attn.in_proj_weight[self.attn.q.linear.out_features:2*self.attn.q.linear.out_features, :])
                self.attn.k.linear.bias.copy_(original_block.attn.in_proj_bias[self.attn.q.linear.out_features:2*self.attn.q.linear.out_features])

                self.attn.v.linear.weight.copy_(original_block.attn.in_proj_weight[2*self.attn.q.linear.out_features:3*self.attn.q.linear.out_features, :])
                self.attn.v.linear.bias.copy_(original_block.attn.in_proj_bias[2*self.attn.q.linear.out_features:3*self.attn.q.linear.out_features])

                self.out_proj.linear.weight.copy_(original_block.attn.out_proj.weight)
                self.out_proj.linear.bias.copy_(original_block.attn.out_proj.bias)

            else:
                raise TypeError(f"Unsupported attention module type in original_block: {type(original_block.attn)}")


        # After loading pretrained weights, re-initialize LoRA-specific parameters
        # This ensures that at the start of finetuning, the LoRA adjustment is zero.
        self.attn.reset_parameters()
        self.out_proj.reset_parameters()
        self.mlp.c_fc.reset_parameters()
        self.mlp.c_proj.reset_parameters()

        print("Successfully loaded weights from original ResidualAttentionBlock and reset LoRA parameters.")




class LoRALinear(nn.Module):
    def __init__(self, linear_layer, rank, lora_alpha, lora_dropout=DROPOUT_P):
        super().__init__()
        self.linear = linear_layer
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.rank = rank
        self.lora_alpha = lora_alpha
        
        #  LoRA dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = nn.Identity()
            
        # Now, only one set of A and B matrices, shared for all tasks
        self.lora_A = nn.Parameter(torch.zeros(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        
        # Scaling factor
        if self.lora_alpha > 0:
            self.scaling = self.lora_alpha / self.rank
        else:
            self.scaling = 1.0

        self.reset_parameters()

    def reset_parameters(self):
        # Freeze the original linear layer
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False
            
        # Initialize LoRA matrices
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        result = self.linear(x)
        
        x_dropout = self.lora_dropout(x)
        
        delta = (x_dropout @ self.lora_A.T @ self.lora_B.T) * self.scaling
        result += delta
            
        return result

def add_lora_to_backbone(backbone_module, rank, lora_alpha):
    """
    Recursively replaces nn.Linear layers in specified blocks of the backbone
    with LoRALinear layers, using shared LoRA weights.
    """
    target_layer_names = ['in_proj', 'out_proj', 'c_fc', 'c_proj']
    
    for name, module in backbone_module.named_children():
        if isinstance(module, nn.Linear) and name in target_layer_names:
            # Replace the Linear layer with the shared LoRALinear
            lora_layer = LoRALinear(module, rank, lora_alpha)
            setattr(backbone_module, name, lora_layer)
        elif len(list(module.children())) > 0:
            # Recurse into sub-modules
            add_lora_to_backbone(module, rank, lora_alpha)