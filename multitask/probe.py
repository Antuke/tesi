import argparse
import sys
import os
import types
from typing import List, Optional, Dict, Tuple
from abc import ABC, abstractmethod
from enum import Enum

from dotenv import load_dotenv
load_dotenv()
sys.path.append(os.getenv("REPO_PATH"))

from dotenv import load_dotenv
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
import torch
import torch.nn as nn


from core.vision_encoder import pe 
from multitask.moe_task_aware import MoELayerTaskAware, ExpertPe, ExpertSiglip
from utils.commons import *


DROPOUT_P = 0.1
GENDERS_NUM = 2
EMOTIONS_NUM = 7
AGE_GROUPS_NUM = 9
DEVICE = 'cuda' 

class BackboneType(Enum):
    PE = "pe"
    SIGLIP = "siglip"


class BackboneStrategy(ABC):
    """Abstract base class for backbone-specific logic """
    def __init__(self, backbone: nn.Module, num_tasks: int):
        self.backbone = backbone
        self.num_tasks = num_tasks

    @abstractmethod
    def unfreeze_layers(self, num_layers_to_unfreeze: int):
        pass

    @abstractmethod
    def enable_moe(self, num_experts: int, top_k: int):
        pass

    @abstractmethod
    def enable_k_probes(self):
        pass
    
    @abstractmethod
    def get_probe(self) -> nn.Parameter:
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Dict]]:
        pass

class PEStrategy(BackboneStrategy):
    """Strategy for the PE (VisionTransformer) backbone."""
    def unfreeze_layers(self, num_layers_to_unfreeze: int):
        # Unfreezing logic is now encapsulated here
        for param in self.backbone.attn_pool.parameters():
            param.requires_grad = True
        if num_layers_to_unfreeze > 0:
            self.backbone.proj.requires_grad = True
            for param in self.backbone.ln_post.parameters():
                param.requires_grad = True
            layers = self.backbone.transformer.resblocks
            for block in layers[-min(len(layers), num_layers_to_unfreeze):]:
                for param in block.parameters():
                    param.requires_grad = True

    def enable_moe(self, num_experts: int, top_k: int):
        self.backbone.attn_pool = convert_pe_pooling_to_moe(
            self.backbone.attn_pool, self.num_tasks, num_experts=num_experts, top_k=top_k
        )

    def enable_k_probes(self):
        original_probe = self.backbone.attn_pool.probe.data
        new_probes = original_probe.repeat(1, self.num_tasks, 1)
        self.backbone.attn_pool.probe = nn.Parameter(new_probes)

    def get_probe(self) -> nn.Parameter:
        return self.backbone.attn_pool.probe.data

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Dict]]:
        # Assumes the MoE layer, if enabled, returns the balancing loss
        # The 'modified' flag was ambiguous, simplifying to a consistent output format
        output = self.backbone(x, modified=True)
        if isinstance(output, tuple): # MoE case
            shared_features, balancing_loss, stats = output
            return shared_features, balancing_loss, stats
        else: # Standard case
            return output, None, None


class _SigLIPKProbeHead(nn.Module):
    """
    This module wraps the original SigLIP head layers to return all probe outputs.
    """
    def __init__(self, original_head: nn.Module):
        super().__init__()
        # Copy necessary layers and parameters from the original head
        self.probe = original_head.probe
        self.attention = original_head.attention
        self.layernorm = original_head.layernorm
        self.mlp = original_head.mlp

    def forward(self, hidden_state: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = hidden_state.shape[0]
        # The k-probes are already part of self.probe, so we just repeat for the batch
        probe = self.probe.repeat(batch_size, 1, 1)


        if attention_mask is not None:
            raise ValueError("Custom SigLIP k-probe head does not support attention_mask.")

        hidden_state = self.attention(probe, hidden_state, hidden_state)[0]
        residual = hidden_state
        hidden_state = self.layernorm(hidden_state)
        hidden_state = residual + self.mlp(hidden_state)
        
        # Return the entire tensor of shape [batch_size, num_probes, hidden_dim]
        return hidden_state

class _SigLIPKMoeHead(nn.Module):
    """
    A wrapper class that converts a standard Siglip2 pooling head into a
    multi-task MoE-based head.
    """
    def __init__(
        self, 
        original_pooler: nn.Module, # Expects a Siglip2MultiheadAttentionPoolingHead
        num_tasks: int,
        num_experts: int,
        top_k: int
    ):
        super().__init__()
        

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
            expert_class = ExpertSiglip
        )
        
        # Seed the new experts with the weights from the original MLP
        original_mlp_weights = original_pooler.mlp.state_dict()
        for i in range(num_experts):
            self.mlp.experts[i].fc1.weight.data.copy_(original_mlp_weights['fc1.weight'])
            self.mlp.experts[i].fc1.bias.data.copy_(original_mlp_weights['fc1.bias'])
            self.mlp.experts[i].fc2.weight.data.copy_(original_mlp_weights['fc2.weight'])
            self.mlp.experts[i].fc2.bias.data.copy_(original_mlp_weights['fc2.bias'])
  


    def forward(self, hidden_state: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """
        The new forward pass that handles MoE logic.
        """
        batch_size = hidden_state.shape[0]
        probe = self.probe.repeat(batch_size, 1, 1)

        if attention_mask is not None:
            raise ValueError("Custom SigLIP k-probe head does not support attention_mask.")

        attn_output = self.attention(probe, hidden_state, hidden_state, attn_mask=attention_mask)[0]

        residual = attn_output
        layernorm_output = self.layernorm(attn_output)
        
        moe_output, moe_loss, moe_stats = self.mlp(layernorm_output)
        
        hidden_state = residual + moe_output
        
        return hidden_state, moe_loss, moe_stats


class SigLIPStrategy(BackboneStrategy):
    """Strategy for the SigLIP backbone."""
    def unfreeze_layers(self, num_layers_to_unfreeze: int):
        for param in self.backbone.head.parameters():
            param.requires_grad = True
        if num_layers_to_unfreeze > 0:
            for p in self.backbone.post_layernorm.parameters():
                p.requires_grad = True
            layers = self.backbone.encoder.layers
            for l in layers[-min(len(layers), num_layers_to_unfreeze):]:
                for param in l.parameters():
                    param.requires_grad = True

    def enable_moe(self, num_tasks, num_experts: int, top_k: int):
        self.backbone.head = _SigLIPKMoeHead(self.backbone.head, num_tasks=num_tasks, num_experts=num_experts, top_k=top_k)


    def enable_k_probes(self):
        original_probe = self.backbone.head.probe.data
        new_probes = original_probe.repeat(1, self.num_tasks, 1)
        self.backbone.head.probe = nn.Parameter(new_probes)
        self.backbone.head = _SigLIPKProbeHead(self.backbone.head)

    def get_probe(self) -> nn.Parameter:
        return self.backbone.head.probe.data

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Dict]]:
        output = self.backbone(pixel_values=x).pooler_output
        if isinstance(output, tuple): # MoE case
            shared_features, balancing_loss, stats = output
            return shared_features, balancing_loss, stats
        else: # Standard case
            return output, None, None

def _get_classifier_head(in_dim: int, out_dim: int, dropout_p: float = 0.0) -> nn.Sequential:
    """Creates a classifier head with dropout and a linear layer."""
    return nn.Sequential(
        nn.Dropout(p=dropout_p),
        nn.Linear(in_dim, out_dim)
    )

def backbone_strategy_factory(backbone: nn.Module, num_tasks: int) -> BackboneStrategy:
    """Factory function to select the appropriate strategy."""
    if isinstance(backbone, pe.VisionTransformer): # PE backbone
        return PEStrategy(backbone, num_tasks)
    else: # SigLIP backbone TODO add isinstance
        return SigLIPStrategy(backbone, num_tasks)



class MultiTaskProbe(nn.Module):
    """
    A unified multi-task learning probe that delegates backbone-specific logic
    to strategy classes for improved modularity and maintainability.
    """
    def __init__(self,
                 backbone: nn.Module,
                 backbone_output_dim: int,
                 tasks: Dict[str, int], 
                 num_layers_to_unfreeze: int = 0,
                 use_k_probes: bool = False,
                 use_moe: bool = False,
                 moe_num_experts: int = 8,
                 moe_top_k: int = 2):

        super().__init__()
        
        if use_k_probes and use_moe:
            raise ValueError("use_k_probes and use_moe cannot be True at the same time.")

        self.backbone = backbone
        self.tasks = tasks
        self.num_tasks = len(tasks)
        self.use_k_probes = use_k_probes
        self.use_moe = use_moe
        print(self.tasks.items())
        self.heads = nn.ModuleDict({
            task_name: _get_classifier_head(backbone_output_dim, out_dim, DROPOUT_P)
            for task_name, out_dim in self.tasks.items()
        })
        self.strategy = backbone_strategy_factory(self.backbone, self.num_tasks)      
        self._setup_layers(num_layers_to_unfreeze, moe_num_experts, moe_top_k)

    def _setup_layers(self, num_layers_to_unfreeze: int, moe_num_experts: int, moe_top_k: int):
        """Freezes backbone and delegates setup to the selected strategy."""
        for param in self.backbone.parameters():
            param.requires_grad = False

        if self.use_moe:
            self.strategy.enable_moe(self.num_tasks,moe_num_experts, moe_top_k)
        elif self.use_k_probes:
            self.strategy.enable_k_probes()
            
        self.strategy.unfreeze_layers(num_layers_to_unfreeze)

    def forward(self, x: torch.Tensor):
        """Forward pass through the model."""
        shared_features, balancing_loss, stats = self.strategy.forward(x)

        # If using k-probes or MoE, we expect per-task embeddings
        if self.use_k_probes or self.use_moe:
            # [batch_size, num_tasks, feature_dim]
            logits = [
                self.heads[task_name](shared_features[:, i, :])
                for i, task_name in enumerate(self.tasks.keys())
            ]
        else:
            # Features are shared: [batch_size, feature_dim]
            logits = [
                head(shared_features)
                for _, head in self.heads.items()
            ]

        if self.use_moe:
            return logits, balancing_loss, stats
        return logits


    def load_heads(self, ckpt_paths: Dict[str, str], device: str = 'cuda'):
        """Loads weights from checkpoints into the respective heads."""
        for task_name, head in self.heads.items():
             if ckpt_path := ckpt_paths.get(task_name):
                try:
                    state_dict = torch.load(ckpt_path, map_location=device)
                    # The head is a Sequential(Dropout, Linear). We load into the Linear layer.
                    head[1].load_state_dict(state_dict)
                    print(f"Successfully loaded weights for '{task_name}' head from {ckpt_path}")
                except FileNotFoundError:
                    print(f"ERROR: Head checkpoint not found for '{task_name}' at '{ckpt_path}'.")
                except Exception as e:
                    print(f"An error occurred while loading '{task_name}' head: {e}")
             else:
                print(f"No checkpoint provided for '{task_name}' head. It remains randomly initialized.")

    def save(self, path: str, epoch: int, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler):
        """Saves a checkpoint with trainable weights and optimizer state."""
        try:
            # Filter for parameters that require gradients
            trainable_state_dict = {name: param for name, param in self.named_parameters() if param.requires_grad}
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': trainable_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()
            }
            torch.save(checkpoint, path)
            print(f"Successfully saved multi-task checkpoint to: {path}")
        except Exception as e:
            print(f"Error saving multi-task checkpoint to {path}: {e}")

    def load(self, path: str, optimizer: Optional[torch.optim.Optimizer] = None, scheduler: Optional[torch.optim.lr_scheduler] = None, device: str = 'cuda'):
        """Loads a multi-task checkpoint into the model and optimizer."""
        try:
            checkpoint = torch.load(path, map_location=device)
            # Use strict=False to only load parameters present in the checkpoint
            self.load_state_dict(checkpoint['model_state_dict'], strict=False)
            if optimizer and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scheduler and 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(f"Successfully loaded multi-task checkpoint from: {path}")
            return checkpoint.get('epoch', 0)
        except FileNotFoundError:
            print(f"Error: Multi-task checkpoint file not found at '{path}'")
            return 0
        except Exception as e:
            print(f"An error occurred while loading the multi-task checkpoint: {e}")
            return 0

    def log_probe_similarity(self, file_path: str = "probe_similarity.jpg", title: str = "Probe Similarity Matrix"):
        """Calculates and plots the cosine similarity matrix of the probes."""
        if not self.use_k_probes and not self.use_moe:
            print("Probe similarity logging is only available for k-probes or MoE setups.")
            return
            
        directory = os.path.dirname(file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        self.eval()
        with torch.no_grad():
            probes = self.strategy.get_probe().squeeze(0)
            probes = F.normalize(probes, p=2, dim=1)
            similarity_matrix = torch.matmul(probes, probes.T).cpu().numpy()

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(similarity_matrix, ax=ax, annot=True, cmap='viridis', fmt='.2f', vmin=0, vmax=1)
        ax.set_title(title, fontsize=16)
        ax.set_xlabel("Probe Index", fontsize=12)
        ax.set_ylabel("Probe Index", fontsize=12)
        plt.savefig(file_path, bbox_inches='tight', dpi=150)
        plt.close(fig)

if __name__ == '__main__':
    print("--- Testing MoE Multitask Probe ---")
    device = 'cuda'
    tasks = {'age': AGE_GROUPS_NUM, 'gender': GENDERS_NUM, 'emotion': EMOTIONS_NUM}
    backbone_moe, _, hidden_size_moe = get_backbone('google/Siglip2-base-patch16-224')
    model_moe = MultiTaskProbe(
        backbone=backbone_moe,
        backbone_output_dim=hidden_size_moe,
        tasks=tasks,
        use_moe=True
    ).to(device)
    x_moe = torch.randn(10, 3, 224, 224, device=device)
    logits_list_moe, balancing_loss, gate_stats = model_moe(x_moe)
    for i, (task_name, logits) in enumerate(zip(tasks.keys(), logits_list_moe)):
        print(f"Logits for '{task_name}' have shape: {logits.shape}")
    print(f"Balancing loss: {balancing_loss.item()}")
    print(f"gate stats = {gate_stats}")
    print("-" * 20)