"""This files contains the concrete strategies and factory for those strategies used by the MultiTaskProbe class
"""


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
from peft import PeftModel, get_peft_model, LoraConfig, TaskType

from core.vision_encoder import pe 
from utils.commons import *
from multitask.wrappers import SigLIPKMoeHead, SigLIPKProbeHead, PEMoeViT

DROPOUT_P = 0.5
DEVICE = 'cuda' 



def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"[TRAIANBLE]\ntrainable params: {trainable_params} || all params: {all_param} || "
        f"trainable%: {100 * trainable_params / all_param:.2f}"
    )


    
class BackboneType(Enum):
    PE = "pe"
    SIGLIP = "siglip"

class BackboneStrategy(nn.Module, ABC):
    """Abstract base class for backbone-specific logic """
    def __init__(self, backbone: nn.Module, num_tasks: int, task_agnostic_gate: bool, use_lora : bool):
        super().__init__()
        self.backbone = backbone
        self.num_tasks = num_tasks
        self.task_agnostic_gate = task_agnostic_gate
        self.use_lora = use_lora

    def setup_layers(self, num_layers_to_unfreeze: int):
        for param in self.parameters():
            param.requires_grad = False

        if self.use_lora:
            print("[LORA] Adding LoRa Matrices")
            self.lora_config_init()
            self.backbone = get_peft_model(self.backbone, self.lora_config)
        """ 
        target_weight_name = 'base_model.model.transformer.resblocks.10.attn.out_proj.lora_A.default.weight'
        self.found_param = None
        for name, param in self.backbone.named_parameters():
            if name == target_weight_name:
                self.found_param = param
                print('PARAMETER FOUNDED!')
                break 
        target_weight_name = 'base_model.model.transformer.resblocks.5.mlp.c_fc.lora_A.default.weight'
        self.working_param = None
        for name, param in self.backbone.named_parameters():
            if name == target_weight_name:
                self.working_param = param
                print('PARAMETER FOUNDED!')
                break 
        p_work = self.working_param
        print(f"\n[WORKING PARAM] 'base_model.model.transformer.resblocks.5.mlp.c_fc.lora_A.default.weight'")
        print(f"  - Requires Grad: {p_work.requires_grad}")
        print(f"  - Is Leaf: {p_work.is_leaf}")
        print(f"  - Dtype: {p_work.dtype}")


        # Check the problematic one
        p_prob = self.found_param
        print(f"\n[PROBLEM PARAM]  'base_model.model.transformer.resblocks.10.attn.out_proj.lora_A.default.weight'")
        print(f"  - Requires Grad: {p_prob.requires_grad}")
        print(f"  - Is Leaf: {p_prob.is_leaf}")
        print(f"  - Dtype: {p_prob.dtype}")
        """
    def lora_config_init(self):
        """ Freeze all parameters and unfreeze the specified number of layers. Plus it adds lora matrices if use_lora is True"""
        lora_target_modules = []

        for name, module in self.backbone.named_modules():
            # Check if the module is a Linear layer 
            if isinstance(module, nn.Linear):
                lora_target_modules.append(name)

        # print(f'[LORA] target modules = {lora_target_modules}')
        self.lora_config =  LoraConfig(
            r=16,                
            lora_alpha=32,      
            target_modules=lora_target_modules, 
            lora_dropout=0.1,
            init_lora_weights='eva',
            bias = "none",
        )



    @abstractmethod
    def enable_moe(self, num_experts: int, top_k: int):
        """ Subsitutes tha last dense layer of the attention pooling layer with a MoE layer. It also
        by default also enable the k_probes """
        pass

    @abstractmethod
    def enable_k_probes(self):
        """ Substitutes the attention pooling layer 'query probe' with 'k-query probes', one for each task. 
        If the backbone has a projection layer, it is also replaced with k-projection layers."""
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
        """Unfreezes additional layers and returns ONLY the newly unfrozen parameters. If the attention pooling layer
        was frozen, and num_layers_to_unfreeze is greater or equal to 0, it will be unfrozen and returned also, plus the
        other layers that are required to be unfrozen. This is intended to be used with optimizer.add_param_group()."""
        pass

    @abstractmethod
    def get_parameter_groups(self, num_layers_to_unfreeze :int):
        """ Returns the initial parameter groups for the optimizer. If num_layers_to_unfreeze is 0, only the attention pooling layer is returned.
        if num_layers_to_unfreeze is -1, an empty list is returned (the classification head are returned by the MTLProbe class regardless).
        This is intended to be used with optimizer.add_param_group(). """
        pass

    
class PEStrategy(BackboneStrategy):

    def unfreeze_and_get_new_params(self, target_num_unfrozen_layers: int):
        """
        Unfreezes additional layers and returns ONLY the newly unfrozen parameters.
        This is intended to be used with optimizer.add_param_group().

        Args:
            target_num_unfrozen_layers (int): The target total number of unfrozen transformer layers.

        Returns:
            list: A flat list of newly unfrozen parameters.
        """
        new_param_groups = []



        if target_num_unfrozen_layers <= self.currently_unfrozen_layers:
            print(f"Request to unfreeze {target_num_unfrozen_layers} layers, but {self.currently_unfrozen_layers} are already unfrozen. No new layers to unfreeze.")
            return new_param_groups

        # This block only runs on the very first call when everything is frozen expect the heads.
        if self.currently_unfrozen_layers == -1:

            # we return all the lora params
            if self.use_lora:
                lora_params = []
                for name, param in self.backbone.named_parameters():
                    if 'lora_' in name:
                        if 'attn_pool' not in name:
                            print(f"[LORA] Adding LoRa matrices: {name}")
                            lora_params.append(param)

                    if 'attn_pool' in name:
                        print(f"[LORA] Adding attn_pool weights: {name}")
                        param.requires_grad = True
                        lora_params.append(param)

                if lora_params:
                    new_param_groups.append({'name': 'lora', 'params': lora_params})
                    print(f"Identified 'lora' parameter group with {len(lora_params)} tensors.")
                    print_trainable_parameters(self.backbone)
                self.currently_unfrozen_layers = 99
                return new_param_groups
                

            print("Unfreezing the head parameters (attn_pool and proj)...")
            head_params_to_unfreeze = []
            
            # Unfreeze and collect parameters for attn_pool
            for param in self.backbone.attn_pool.parameters():
                param.requires_grad = True
                head_params_to_unfreeze.append(param)
            
            # Unfreeze and collect parameters for proj
            if self.backbone.proj is not None:
                # Assuming self.backbone.proj is a parameter tensor (nn.Parameter)  
                self.backbone.proj.requires_grad = True
                head_params_to_unfreeze.append(self.backbone.proj)

            if head_params_to_unfreeze:
                new_param_groups.append({
                    'name': 'attn_pool',
                    'params': head_params_to_unfreeze
                })
            
            # Update state to signify the head is now unfrozen.

        if self.use_lora:
            print('[WARNING] When using LORA all the parameters of the LORA are unfrozen, and the rest should stay FROZEN!')
            return []

        # This block runs if the target is greater than the current number of unfrozen layers.
        if target_num_unfrozen_layers > self.currently_unfrozen_layers:
            backbone_params_to_unfreeze = []
            layers = self.backbone.transformer.resblocks
            
            # Determine the slice of layers to unfreeze.
            # Example: If 0 are unfrozen and target is 4, we unfreeze layers[-4:].
            # Example: If 2 are unfrozen and target is 4, we unfreeze layers[-4:-2].
            layers = self.backbone.transformer.resblocks
            start_layer = len(layers) - target_num_unfrozen_layers
            end_layer = len(layers) - self.currently_unfrozen_layers if self.currently_unfrozen_layers > 0 else len(layers)

            # Ensure start_layer is not out of bounds
            start_layer = max(0, start_layer)

            layers_to_unfreeze = layers[start_layer:end_layer]
            
            
            if layers_to_unfreeze:
                print(f"Unfreezing {len(layers_to_unfreeze)} new transformer blocks...")
                for block in layers_to_unfreeze:
                    for param in block.parameters():
                        if param.requires_grad is False:  # Only unfreeze if currently frozen
                            param.requires_grad = True
                            backbone_params_to_unfreeze.append(param)

                if backbone_params_to_unfreeze:
                    new_param_groups.append({
                        'name': f'backbone_{target_num_unfrozen_layers}',
                        'params': backbone_params_to_unfreeze
                    })

        self.currently_unfrozen_layers = target_num_unfrozen_layers
        
        return new_param_groups

    def get_parameter_groups(self, num_layers_to_unfreeze: int):
        """
        Sets the initial state of trainable parameters and returns all of them
        in groups. This is intended for the initial setup of the optimizer.
        """
        named_param_groups = []
        if num_layers_to_unfreeze < 0:
            print("[SETUP] Only classification heads will be trained")
            self.currently_unfrozen_layers = -1
            return named_param_groups
        else:
            self.currently_unfrozen_layers = num_layers_to_unfreeze
        
        if self.use_lora:
            print("[SETUP] Using LoRa!")
            self.currently_unfrozen_layers = 99
            if self.use_lora:
                lora_params = []
                for name, param in self.backbone.named_parameters():
                    # train each lora matrices that is not in the attn_pool
                    if 'lora_' in name:
                        print(f"[LORA] Adding LoRa matrices: {name}")
                        lora_params.append(param)
                    # train every parameter that is not a lora matrices
                    if 'gating' in name or 'probe' in name:
                        if 'lora_' not in name:
                            print(f"[LORA] Adding head weights: {name}")
                            param.requires_grad = True
                            lora_params.append(param)
                    if name == 'base_model.model.proj':
                        print(f"[LORA] Adding head weights: {name}")
                        param.requires_grad = True
                        lora_params.append(param)

                if lora_params:
                    named_param_groups.append({'name': 'lora', 'params': lora_params})
                    print(f"[LORA] Identified 'lora' parameter group with {len(lora_params)} tensors.")
                    print_trainable_parameters(self.backbone)
                return named_param_groups
        
        head_params = []
        # Always make attn_pool and proj trainable from the start
        attn_pool_params = list(self.backbone.attn_pool.parameters())
        for param in attn_pool_params:
            param.requires_grad = True
        head_params.extend(attn_pool_params)

        if self.backbone.proj is not None:
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

        
        return named_param_groups

    def enable_moe(self, num_tasks: int, num_experts: int, top_k: int):
        self.backbone = PEMoeViT(self.backbone, num_tasks=num_tasks, num_experts=num_experts, top_k=top_k, task_agnostic_gate=self.task_agnostic_gate)
    
    def enable_k_probes(self):
        original_probe = self.backbone.attn_pool.probe.data
        new_probes = original_probe.repeat(1, self.num_tasks, 1)
        self.backbone.attn_pool.probe = nn.Parameter(new_probes)
        if self.backbone.proj is not None:
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
        if len(output) == 4: # MoE case
            shared_features, balancing_loss, stats, attn_weights = output
            return shared_features, balancing_loss, stats, attn_weights
        else: 
            shared_features = output
            return shared_features, None, None, None

    def get_last_shared_layer(self):
        return self.backbone.transformer.resblocks[-1].mlp.c_proj

class SigLIPStrategy(BackboneStrategy):

       

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
        if len(output) == 4: # MoE case
            shared_features, balancing_loss, stats, attn_weights = output
            return shared_features, balancing_loss, stats, attn_weights
        if len(output) == 2: # K-probes
            shared_features, attn_weights = output
            return shared_features, None, None, attn_weights

    def get_last_shared_layer(self) -> nn.Module:
        return self.backbone.encoder.layers[-1]


    def unfreeze_and_get_new_params(self, target_num_unfrozen_layers: int):
        new_param_groups = []

        # Guard clause: Check if we actually need to unfreeze anything new.
        if target_num_unfrozen_layers <= self.currently_unfrozen_layers:
            print(f"Request to unfreeze {target_num_unfrozen_layers} layers, but {self.currently_unfrozen_layers} are already unfrozen. No new layers to unfreeze.")
            return new_param_groups

        # This block only runs on the very first call when everything is frozen (state is -1).
        if self.currently_unfrozen_layers == -1:

            if self.use_lora:
                lora_params = []
                for name, param in self.backbone.named_parameters():
                    if 'lora_' in name:
                        if 'attn_pool' not in name:
                            print(f"[LORA] Adding LoRa matrices: {name}")
                            lora_params.append(param)

                    if 'head' in name or 'base_model.model.head.proj' == name:
                        lora_params.append(param)

                if lora_params:
                    new_param_groups.append({'name': 'lora', 'params': lora_params})
                    print(f"Identified 'lora' parameter group with {len(lora_params)} tensors.")
                    print_trainable_parameters(self.backbone)
                
                return new_param_groups

            print("Unfreezing the SigLip head parameters...")
            head_params_to_unfreeze = []
            
            # Unfreeze and collect all parameters from the head module
            for param in self.backbone.head.parameters():
                param.requires_grad = True
                head_params_to_unfreeze.append(param)

            if head_params_to_unfreeze:
                new_param_groups.append({
                    'name': 'attn_pool',
                    'params': head_params_to_unfreeze
                })
            self.currently_unfrozen_layers = 0
            return new_param_groups
            
            # Update state: the head is unfrozen, but 0 encoder layers are.

        if self.use_lora:
            print('[WARNING] When using LORA all the parameters of the LORA are unfrozen, and the rest should stay FROZEN!')
            return []

        # This block runs if the target number of layers is greater than what's currently unfrozen.
        if target_num_unfrozen_layers > self.currently_unfrozen_layers:
            backbone_params_to_unfreeze = []
            layers = self.backbone.encoder.layers
            
            # Determine the correct slice of layers to unfreeze
            start_index = -target_num_unfrozen_layers
            end_index = -self.currently_unfrozen_layers if self.currently_unfrozen_layers > 0 else None
            
            layers_to_unfreeze = layers[start_index:end_index]
            if layers_to_unfreeze:
                print(f"Unfreezing {len(layers_to_unfreeze)} new SigLIP transformer layers...")
                for layer in layers_to_unfreeze:
                    for param in layer.parameters():
                        param.requires_grad = True
                        backbone_params_to_unfreeze.append(param)

                if backbone_params_to_unfreeze:
                    new_param_groups.append({
                        'name': f'backbone_{target_num_unfrozen_layers}',
                        'params': backbone_params_to_unfreeze
                    })

        self.currently_unfrozen_layers = target_num_unfrozen_layers

        return new_param_groups

    def get_parameter_groups(self, num_layers_to_unfreeze: int):
        """
        Sets the initial state of trainable parameters for the SigLIP2 backbone and returns them in groups.
        """
        named_param_groups = []
        self.currently_unfrozen_layers = num_layers_to_unfreeze

        if num_layers_to_unfreeze < 0:
            print("[SETUP] No layers to unfreeze! Only head classifcation heads will be trained!")
            return named_param_groups  # No layers to unfreeze, return empty list
        
        if self.use_lora:
            print("[SETUP] Using LoRa!")
            self.currently_unfrozen_layers = 99
            if self.use_lora:
                lora_params = []
                for name, param in self.backbone.named_parameters():
                    if 'lora_' in name:
                        print(f"[LORA] Adding LoRa matrices: {name}")
                        lora_params.append(param)

                    if 'probe' in name or 'gating' in name or name == 'base_model.model.head.proj':
                        print(f"[LORA] Adding {name}")
                        lora_params.append(param)

                if lora_params:
                    named_param_groups.append({'name': 'lora', 'params': lora_params})
                    print(f"Identified 'lora' parameter group with {len(lora_params)} tensors.")
                    print_trainable_parameters(self.backbone)
                return named_param_groups
            

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
        
        return named_param_groups

    
    
def backbone_strategy_factory(backbone: nn.Module, num_tasks: int, task_agnostic_gate : bool, use_lora : bool) -> BackboneStrategy:
    """Factory function to select the appropriate strategy."""
    if isinstance(backbone, pe.VisionTransformer): # PE backbone
        return PEStrategy(backbone, num_tasks, task_agnostic_gate, use_lora)
    else: # SigLIP backbone TODO add isinstance
        return SigLIPStrategy(backbone, num_tasks, task_agnostic_gate, use_lora)