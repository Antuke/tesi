"""This files contains the MultiTaskProbe class, that is configured to handle the three multi-task probing approach 
that have been experimented in this project, using the Strategy and factory pattern."""

import sys
import os
import types
from typing import Optional, Dict, Tuple
from abc import ABC, abstractmethod
from enum import Enum

from dotenv import load_dotenv
load_dotenv()
sys.path.append(os.getenv("REPO_PATH"))

import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
import torch
import torch.nn as nn


from core.vision_encoder import pe 
from utils.commons import *
from multitask.wrappers import SigLIPKMoeHead, SigLIPKProbeHead, PEMoeViT

DROPOUT_P = 0.5
DEVICE = 'cuda' 

class BackboneType(Enum):
    PE = "pe"
    SIGLIP = "siglip"


class BackboneStrategy(ABC):
    """Abstract base class for backbone-specific logic """
    def __init__(self, backbone: nn.Module, num_tasks: int, task_agnostic_gate: bool):
        self.backbone = backbone
        self.num_tasks = num_tasks
        self.task_agnostic_gate = task_agnostic_gate

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
    @abstractmethod
    def get_last_shared_layer(self) -> nn.Module:
        pass
    
    @abstractmethod
    def unfreeze_and_get_new_params(self, num_layers_to_unfreeze : int):
        pass

class PEStrategy(BackboneStrategy):
    """Strategy for the PE (VisionTransformer) backbone."""
    def unfreeze_layers(self, num_layers_to_unfreeze: int):
        
        named_param_groups = []

        head_params = []
        attn_pool_params = list(self.backbone.attn_pool.parameters())
        for param in attn_pool_params:
            param.requires_grad = True
        head_params.extend(attn_pool_params)

        if self.backbone.proj is not None:
            self.backbone.proj.requires_grad = True
            head_params.append(self.backbone.proj)

        named_param_groups.append({'name': 'head', 'params': head_params})
        print("Identified 'head' parameter group for PE.")

        if num_layers_to_unfreeze > 0:
            backbone_params = []
            # Unfreeze post-layernorm
            ln_post_params = list(self.backbone.ln_post.parameters())
            for param in ln_post_params:
                param.requires_grad = True
            backbone_params.extend(ln_post_params)

            # Unfreeze transformer blocks
            layers = self.backbone.transformer.resblocks
            num_to_unfreeze = min(len(layers), num_layers_to_unfreeze)
            print(f"Unfreezing the last {num_to_unfreeze} PE transformer blocks.")
            for block in layers[-num_to_unfreeze:]:
                block_params = list(block.parameters())
                for param in block_params:
                    param.requires_grad = True
                backbone_params.extend(block_params)

            if backbone_params:
                named_param_groups.append({'name': 'backbone', 'params': backbone_params})
                print("Identified 'backbone' parameter group for PE.")

        return named_param_groups

    def unfreeze_and_get_new_params(self, target_num_unfrozen_layers: int):
        """
        Unfreezes additional layers and returns ONLY the newly unfrozen parameters.
        This is intended to be used with optimizer.add_param_group().
        """
        # Check if we actually need to unfreeze anything new
        if target_num_unfrozen_layers <= self.currently_unfrozen_layers:
            print(f"Request to unfreeze {target_num_unfrozen_layers} layers, but {self.currently_unfrozen_layers} are already unfrozen. No new layers to unfreeze.")
            return []

        new_backbone_params = []
        layers = self.backbone.transformer.resblocks
        
        # Determine the slice of layers to unfreeze
        # Example: If 2 are unfrozen and target is 4, we need layers[-4:-2]
        start_index = -target_num_unfrozen_layers
        end_index = -self.currently_unfrozen_layers if self.currently_unfrozen_layers > 0 else None
        
        layers_to_unfreeze = layers[start_index:end_index]
        
        num_newly_unfrozen = len(layers_to_unfreeze)
        print(f"Unfreezing {num_newly_unfrozen} new transformer blocks...")

        for block in layers_to_unfreeze:
            for param in block.parameters():
                if not param.requires_grad: 
                    param.requires_grad = True
                    new_backbone_params.append(param)
        
        # Update the state
        self.currently_unfrozen_layers = target_num_unfrozen_layers

        if not new_backbone_params:
            return []

        # Return the new parameters in the required group format
        return [{'name': 'backbone', 'params': new_backbone_params}]

    # TODO TEST
    def get_parameter_groups(self, num_layers_to_unfreeze: int):
        """
        Sets the initial state of trainable parameters and returns all of them
        in groups. This is intended for the initial setup of the optimizer.
        """
        # This logic is almost identical to your original method
        named_param_groups = []

        # --- Head Parameters ---
        head_params = []
        # Always make attn_pool and proj trainable from the start
        attn_pool_params = list(self.backbone.attn_pool.parameters())
        for param in attn_pool_params:
            param.requires_grad = True
        head_params.extend(attn_pool_params)

        if self.backbone.proj is not None:
            # Assuming proj is a single parameter tensor, not a module
            self.backbone.proj.requires_grad = True
            head_params.append(self.backbone.proj)
        
        named_param_groups.append({'name': 'head', 'params': head_params})
        print("Identified 'head' parameter group for initial setup.")

        # --- Backbone Parameters ---
        if num_layers_to_unfreeze > 0:
            backbone_params = []
            # Unfreeze post-layernorm
            ln_post_params = list(self.backbone.ln_post.parameters())
            for param in ln_post_params:
                param.requires_grad = True
            backbone_params.extend(ln_post_params)

            # Unfreeze transformer blocks
            layers = self.backbone.transformer.resblocks
            num_to_unfreeze = min(len(layers), num_layers_to_unfreeze)
            print(f"Initially unfreezing the last {num_to_unfreeze} PE transformer blocks.")
            
            for block in layers[-num_to_unfreeze:]:
                for param in block.parameters():
                    param.requires_grad = True
                backbone_params.extend(list(block.parameters()))

            if backbone_params:
                named_param_groups.append({'name': 'backbone', 'params': backbone_params})
                print("Identified 'backbone' parameter group for initial setup.")

        
        self.currently_unfrozen_layers = num_layers_to_unfreeze
        return named_param_groups

    def enable_moe(self, num_tasks: int, num_experts: int, top_k: int):
        self.backbone = PEMoeViT(self.backbone, num_tasks=num_tasks, num_experts=num_experts, top_k=top_k, task_agnostic_gate=self.task_agnostic_gate)

    def enable_k_probes(self):
        original_probe = self.backbone.attn_pool.probe.data
        new_probes = original_probe.repeat(1, self.num_tasks, 1)
        self.backbone.attn_pool.probe = nn.Parameter(new_probes)
        if self.backbone.proj:
            self.backbone.proj = nn.Parameter(torch.stack([self.backbone.proj.clone() for _ in range(self.num_tasks)], dim=0)) 

        def forward(self, x: torch.Tensor, **kwargs):
            x = self.forward_features(x, norm=True, **kwargs)
            x = self._pool(x)

            if self.proj_dim is not None:
                x = torch.einsum('bnd,ndp->bnp', x, self.proj)

            return x

        self.backbone.forward = types.MethodType(forward, self.backbone)

    def get_probe(self) -> nn.Parameter:
        return self.backbone.attn_pool.probe.data

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Dict]]:
        output = self.backbone(x)
        if isinstance(output, tuple): # MoE case
            shared_features, balancing_loss, stats = output
            return shared_features, balancing_loss, stats
        else: 
            return output, None, None

    def get_last_shared_layer(self):
        return self.backbone.ln_post

class SigLIPStrategy(BackboneStrategy):
    """Strategy for the SigLIP backbone."""
    def unfreeze_layers(self, num_layers_to_unfreeze: int):
        named_param_groups = []

        # Group 1: The final classification head
        head_params = list(self.backbone.head.parameters())
        for param in head_params:
            param.requires_grad = True
        named_param_groups.append({'name': 'head', 'params': head_params})
        print("Identified 'head' parameter group for SigLIP.")

        if num_layers_to_unfreeze > 0:
            backbone_params = []
            # Unfreeze post-layernorm
            post_ln_params = list(self.backbone.post_layernorm.parameters())
            for p in post_ln_params:
                p.requires_grad = True
            backbone_params.extend(post_ln_params)

            # Unfreeze transformer layers
            layers = self.backbone.encoder.layers
            num_to_unfreeze = min(len(layers), num_layers_to_unfreeze)
            print(f"Unfreezing the last {num_to_unfreeze} SigLIP transformer layers.")
            for l in layers[-num_to_unfreeze:]:
                layer_params = list(l.parameters())
                for param in layer_params:
                    param.requires_grad = True
                backbone_params.extend(layer_params)

            if backbone_params:
                named_param_groups.append({'name': 'backbone', 'params': backbone_params})
                print("Identified 'backbone' parameter group for SigLIP.")

        return named_param_groups

    def enable_moe(self, num_tasks, num_experts: int, top_k: int):
        self.backbone.head = SigLIPKMoeHead(self.backbone.head, num_tasks=num_tasks, num_experts=num_experts, top_k=top_k, task_agnostic_gate=self.task_agnostic_gate)


    def enable_k_probes(self):
        original_probe = self.backbone.head.probe.data
        new_probes = original_probe.repeat(1, self.num_tasks, 1)
        self.backbone.head.probe = nn.Parameter(new_probes)
        self.backbone.head = SigLIPKProbeHead(self.backbone.head)

    def get_probe(self) -> nn.Parameter:
        return self.backbone.head.probe.data

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Dict]]:
        output = self.backbone(pixel_values=x).pooler_output
        if isinstance(output, tuple): # MoE case
            shared_features, balancing_loss, stats = output
            return shared_features, balancing_loss, stats
        else: 
            return output, None, None

    def get_last_shared_layer(self) -> nn.Module:
        return self.backbone.post_layernorm
    
    
def _get_classifier_head(in_dim: int, out_dim: int, dropout_p: float = 0.0) -> nn.Sequential:
    """Creates a classifier head with dropout and a linear layer."""
    return nn.Sequential(
        nn.BatchNorm1d(in_dim), 
        nn.Dropout(p=dropout_p),
        nn.Linear(in_dim, out_dim)
    )

def backbone_strategy_factory(backbone: nn.Module, num_tasks: int, task_agnostic_gate : bool) -> BackboneStrategy:
    """Factory function to select the appropriate strategy."""
    if isinstance(backbone, pe.VisionTransformer): # PE backbone
        return PEStrategy(backbone, num_tasks, task_agnostic_gate)
    else: # SigLIP backbone TODO add isinstance
        return SigLIPStrategy(backbone, num_tasks, task_agnostic_gate)



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
                 moe_top_k: int = 2,
                 task_agnostic_gate: bool = False):

        super().__init__()
        
        if use_k_probes and use_moe:
            raise ValueError("use_k_probes and use_moe cannot be True at the same time.")

        self.backbone = backbone
        self.tasks = tasks
        self.num_tasks = len(tasks)
        self.use_k_probes = use_k_probes
        self.use_moe = use_moe
        self.heads = nn.ModuleDict({
            task_name: _get_classifier_head(backbone_output_dim, out_dim, DROPOUT_P)
            for task_name, out_dim in self.tasks.items()
        })
        self.strategy = backbone_strategy_factory(self.backbone, self.num_tasks, task_agnostic_gate)      
        self._setup_layers(num_layers_to_unfreeze, moe_num_experts, moe_top_k)
        # used for Uncertainty Weights
        self.log_var = nn.Parameter(torch.zeros(self.num_tasks))
        
        # used for gradNorm
        self.loss_weights = nn.Parameter(torch.ones(self.num_tasks))

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

    def unfreeze_layers(self, layers_to_unfreeze : int):
        return self.strategy.unfreeze_layers(layers_to_unfreeze)

    def load_heads(self, ckpt_paths: Dict[str, str], device: str = 'cuda'):
        """Loads weights from checkpoints into the respective heads."""
        for task_name, head in self.heads.items():
             if ckpt_path := ckpt_paths.get(task_name):
                try:
                    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
                    model_state_dict = state_dict['model_state_dict']
                    new_state_dict = {}
                    for key, value in model_state_dict.items():
                        if key.startswith('linear.1.'):
                            new_key = key[len('linear.1.'):]
                            new_state_dict[new_key] = value
                        else:
                            new_state_dict[key] = value

                    # The head is a Sequential(Dropout, Linear). We load into the Linear layer.
                    head[2].load_state_dict(new_state_dict)
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
            model_buffers = {name: buf for name, buf in self.named_buffers()}
            state_to_save = {**trainable_state_dict, **model_buffers}
            checkpoint = {
                'epoch': epoch,
                'model_state_dict':  state_to_save,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()
            }
            torch.save(checkpoint, path)
            print(f"Successfully saved multi-task checkpoint to: {path}")
        except Exception as e:
            print(f"Error saving multi-task checkpoint to {path}: {e}")

    def load(self, path: str, optimizer: Optional[torch.optim.Optimizer] = None, scheduler = None, device: str = 'cuda'):
        """Loads a multi-task checkpoint into the model and optimizer."""
        try:
            checkpoint = torch.load(path, map_location=device, weights_only=True)
            # Use strict=False to only load parameters present in the checkpoint
            self.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print("--- Layers found in checkpoint ---")
            for layer_name in checkpoint['model_state_dict'].keys():
                print(layer_name)
            print("------------------------------------")
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


DROPOUT_P = 0.2
GENDERS_NUM = 2
EMOTIONS_NUM = 7
AGE_GROUPS_NUM = 9
DEVICE = 'cuda' 
if __name__ == '__main__':
    print("--- Testing MoE Multitask Probe ---")
    device = 'cuda'
    tasks = {'age': AGE_GROUPS_NUM, 'gender': GENDERS_NUM, 'emotion': EMOTIONS_NUM}
    #google/Siglip2-base-patch16-224 /  'PE-Core-B16-224'
    backbone_moe, _, hidden_size_moe = get_backbone('PE-Core-B16-224')
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