import argparse
import sys
import os
from dotenv import load_dotenv
load_dotenv()
sys.path.append(os.getenv("REPO_PATH"))

import torch
import torch.nn as nn
import core.vision_encoder.pe as pe


DROPOUT_P = 0.1
GENDERS_NUM = 1
EMOTIONS_NUM = 7
AGE_GROUPS_NUM = 9


def _get_classifier_head(in_dim, out_dim, dropout_p=0.0):
    return nn.Sequential(
        nn.Dropout(dropout_p),
        nn.Linear(in_dim, out_dim)
    )

class MultiTaskProbe(nn.Module):
    def __init__(self, backbone: nn.Module,
                backbone_output_dim: int,
                num_layers_to_unfreeze: int=0):
        super().__init__()
        self.backbone = backbone 
        self.backbone_type = 'pe' if isinstance(backbone, pe.VisionTransformer) else 'siglip'

        self.gender_head = _get_classifier_head(backbone_output_dim, GENDERS_NUM, DROPOUT_P)
        self.age_head = _get_classifier_head(backbone_output_dim, AGE_GROUPS_NUM, DROPOUT_P)
        self.emotion_head = _get_classifier_head(backbone_output_dim, EMOTIONS_NUM, DROPOUT_P)
        
        for param in self.backbone.parameters():
            param.requires_grad = False 
        
        # PE 
        if hasattr(self.backbone, 'attn_pool'):
            for param in self.backbone.attn_pool.parameters():
                param.requires_grad = True

        # SigLip2
        if hasattr(self.backbone, 'head'):
            for param in self.backbone.head.parameters():
                param.requires_grad = True

        self._setup_layers(num_layers_to_unfreeze)
            
    def _setup_layers(self,num_layers_to_unfreeze):
        """Unfreezes layers that have been trained, and freezes the one that have not to be trained"""
        # first freeze all layers, then unfreeze them if need be
        for param in self.backbone.parameters():
            param.requires_grad = False


        # Attention pooling layer has always to be unfreeze for MTL task
        # PE 
        if hasattr(self.backbone, 'attn_pool'):
            for param in self.backbone.attn_pool.parameters():
                param.requires_grad = True
        # TODO unfreeze logic for pe
        
        # SigLip2
        if hasattr(self.backbone, 'head'):
            for param in self.backbone.head.parameters():
                param.requires_grad = True

            if num_layers_to_unfreeze > 0:
                layers = self.backbone.encoder.layers
                num_to_unfreeze = min(len(layers), num_layers_to_unfreeze)
                layers_to_unfreeze = layers[-num_to_unfreeze:] # get the last layers to unfreeze
                for l in layers_to_unfreeze:
                    for param in l.parameters():
                            param.requires_grad = True



            
            
    def forward(self, x):
        if self.backbone_type == 'siglip':
            shared_features = self.backbone(pixel_values=x).pooler_output # [B, backbone_output_dim]
        else:
            shared_features = self.backbone(x)

        
        gender_logits = self.gender_head(shared_features)
        age_logits = self.age_head(shared_features)
        emotion_logits = self.emotion_head(shared_features)
            
        return gender_logits, age_logits, emotion_logits


    def load_heads(self, gender_ckpt_path=None, age_ckpt_path=None, emotion_ckpt_path=None, device='cpu'):
        """
        Loads the weights from a saved head's state_dict into the respective heads.
        This is now much simpler.
        """
        head_map = {
            'gender': (self.gender_head, gender_ckpt_path),
            'age': (self.age_head, age_ckpt_path),
            'emotion': (self.emotion_head, emotion_ckpt_path)
        }

        for task_name, (head_module, ckpt_path) in head_map.items():
            if ckpt_path is None:
                print(f"No checkpoint provided for '{task_name}' head. It remains randomly initialized.")
                continue
            
            try:
                # Load the state_dict for the nn.Linear layer
                head_state_dict = torch.load(ckpt_path, map_location=device)

                # The linear layer is the second element ([1]) in our nn.Sequential head
                head_module[1].load_state_dict(head_state_dict)
                
                print(f"Successfully loaded weights for '{task_name}' head from {ckpt_path}")

            except FileNotFoundError:
                print(f"ERROR: Head checkpoint file not found for '{task_name}' at '{ckpt_path}'.")
            except Exception as e:
                print(f"An unexpected error occurred while loading '{task_name}' head: {e}")

    def save(self, path: str, epoch: int, optimizer: torch.optim.Optimizer, scheduler : torch.optim.lr_scheduler):
        """Saves a checkpoint containing ONLY the trainable weights and optimizer state."""
        try:
            path = f'{path}/mtl_{self.backbone_type}_{epoch}.pt'
            trainable_state_dict = { name: param for name, param in self.named_parameters() if param.requires_grad }
            checkpoint = {
                'epoch': epoch, 'model_state_dict': trainable_state_dict, 'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()
            }
            torch.save(checkpoint, path)
            print(f"Successfully saved multi-task checkpoint to: {path}")
        except Exception as e:
            print(f"Error saving multi-task checkpoint to {path}: {e}")

    def load(self, path: str, optimizer: torch.optim.Optimizer = None, device: str = 'cuda', scheduler : torch.optim.lr_scheduler = None):
        """Loads a multi-task checkpoint into the model and optimizer."""
        try:
            checkpoint = torch.load(path, map_location=device)
            self.load_state_dict(checkpoint['model_state_dict'], strict=False)
            if optimizer and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scheduler and 'scheduler' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(f"Successfully loaded multi-task checkpoint from: {path}")
            return checkpoint.get('epoch', 0)
        except FileNotFoundError:
            print(f"Error: Multi-task checkpoint file not found at '{path}'")
            return 0
        except Exception as e:
            print(f"An error occurred while loading the multi-task checkpoint: {e}")
            return 0