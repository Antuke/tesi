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
from multitask.strategies import backbone_strategy_factory
DROPOUT_P = 0.5
DEVICE = 'cuda' 




def _get_classifier_head(in_dim: int, out_dim: int, dropout_p: float = 0.0) -> nn.Sequential:
    """Creates a classifier head with dropout and a linear layer."""
    return nn.Sequential(
        nn.BatchNorm1d(in_dim), 
        nn.Dropout(p=dropout_p),
        nn.Linear(in_dim, out_dim)
    )




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
            self.backbone = self.strategy.enable_moe(self.num_tasks,moe_num_experts, moe_top_k)
        elif self.use_k_probes:
            self.backbone = self.strategy.enable_k_probes()
            
        self.strategy.unfreeze_layers(num_layers_to_unfreeze)

    def get_last_shared_layer(self):
        return self.strategy.get_last_shared_layer()


    def forward(self, x: torch.Tensor, return_shared: bool = False) -> dict:
        """
        Forward pass through the model.
        Always returns a dictionary.
        """
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

        # --- output dictionary ---
        outputs = {"logits": logits}

        if self.use_moe:
            outputs["balancing_loss"] = balancing_loss
            outputs["stats"] = stats

        if return_shared:
            outputs["shared_features"] = shared_features
            
        # No more complex logic, just return the dictionary
        return outputs

    def unfreeze_layers(self, layers_to_unfreeze : int):
        return self.strategy.unfreeze_layers(layers_to_unfreeze)

    def unfreeze_and_get_new_params(self, num_layers_to_unfreeze):
        return self.strategy.unfreeze_and_get_new_params(num_layers_to_unfreeze)

    def get_parameter_groups(self, initial_unfrozen_layers,using_uncertainty=False, using_grad_norm=False):
        """Return initial parameter groups, so the attention pooling layer and the classification heads, plus
        the loss weight parameters if need be"""
        param_groups = self.strategy.get_parameter_groups(initial_unfrozen_layers)
        param_groups.append({'name': 'heads', 'params': self.heads.parameters()})
        
        if using_grad_norm:
            param_groups.append({
                'name': 'heads',
                'params': [self.loss_weights], 
            })
        if using_uncertainty:
            param_groups.append({
                'name': 'heads',
                'params': [self.log_var], 
            })

        return param_groups


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

            checkpoint = {
                'epoch': epoch,
                'model_state_dict':  self.state_dict(),
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
            checkpoint = torch.load(path, map_location=device, weights_only=False)
            # Use strict=False to only load parameters present in the checkpoint
            self.load_state_dict(checkpoint['model_state_dict'], strict=True)
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