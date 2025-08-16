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

    @abstractmethod
    def get_parameter_groups(self, num_layers_to_unfreeze :int):
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
        return self.backbone

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
        return self.backbone

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
        return self.backbone.transformer.resblocks[-1].mlp.c_proj

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
        return self.backbone

    def enable_k_probes(self):
        original_probe = self.backbone.head.probe.data
        new_probes = original_probe.repeat(1, self.num_tasks, 1)
        self.backbone.head.probe = nn.Parameter(new_probes)
        self.backbone.head = SigLIPKProbeHead(self.backbone.head)
        return self.backbone
        
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
        return self.backbone.encoder.layers[-1]


    def unfreeze_and_get_new_params(self, target_num_unfrozen_layers: int):
        """
        Unfreezes additional layers of the SigLIP backbone and returns only the newly unfrozen parameters.
        """
        if target_num_unfrozen_layers <= self.currently_unfrozen_layers:
            print(f"Request to unfreeze {target_num_unfrozen_layers} layers, but {self.currently_unfrozen_layers} are already unfrozen. No new layers to unfreeze.")
            return []

        new_backbone_params = []
        layers = self.backbone.encoder.layers

        start_index = -target_num_unfrozen_layers
        end_index = -self.currently_unfrozen_layers if self.currently_unfrozen_layers > 0 else None

        layers_to_unfreeze = layers[start_index:end_index]
        num_newly_unfrozen = len(layers_to_unfreeze)
        print(f"Unfreezing {num_newly_unfrozen} new SigLIP2 transformer layers...")

        for layer in layers_to_unfreeze:
            for param in layer.parameters():
                if not param.requires_grad:
                    param.requires_grad = True
                    new_backbone_params.append(param)
        
        self.currently_unfrozen_layers = target_num_unfrozen_layers

        if not new_backbone_params:
            return []

        return [{'name': 'backbone', 'params': new_backbone_params}]

    def get_parameter_groups(self, num_layers_to_unfreeze: int):
        """
        Sets the initial state of trainable parameters for the SigLIP2 backbone and returns them in groups.
        """
        named_param_groups = []

        # --- Head Parameters ---
        head_params = list(self.backbone.head.parameters())
        for param in head_params:
            param.requires_grad = True
        named_param_groups.append({'name': 'head', 'params': head_params})
        print("Identified 'head' parameter group for SigLIP initial setup.")

        # --- Backbone Parameters ---
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
            print(f"Initially unfreezing the last {num_to_unfreeze} SigLIP transformer layers.")
            
            for l in layers[-num_to_unfreeze:]:
                for param in l.parameters():
                    param.requires_grad = True
                backbone_params.extend(list(l.parameters()))

            if backbone_params:
                named_param_groups.append({'name': 'backbone', 'params': backbone_params})
                print("Identified 'backbone' parameter group for SigLIP initial setup.")
        
        self.currently_unfrozen_layers = num_layers_to_unfreeze
        return named_param_groups

    
    
def backbone_strategy_factory(backbone: nn.Module, num_tasks: int, task_agnostic_gate : bool) -> BackboneStrategy:
    """Factory function to select the appropriate strategy."""
    if isinstance(backbone, pe.VisionTransformer): # PE backbone
        return PEStrategy(backbone, num_tasks, task_agnostic_gate)
    else: # SigLIP backbone TODO add isinstance
        return SigLIPStrategy(backbone, num_tasks, task_agnostic_gate)