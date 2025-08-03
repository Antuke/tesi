import argparse
import sys
import os
from dotenv import load_dotenv
load_dotenv()
sys.path.append(os.getenv("REPO_PATH"))

import torch 
import torch.nn as nn
import sys 
from torch.optim import Optimizer
import core.vision_encoder.pe as pe
DROPOUT_P = 0.1
class Probe(nn.Module):
    """
    A probe to attach to a backbone model.
    """
    def __init__(self, backbone: nn.Module, backbone_output_dim: int, n_out_classes: int, attention_probe: bool):
        super().__init__()

        self.backbone = backbone
        self.linear = nn.Sequential(
            nn.Dropout(p=DROPOUT_P), 
            nn.Linear(backbone_output_dim, n_out_classes)
        )
        self.backbone_type = 'pe' if isinstance(backbone, pe.VisionTransformer) else 'siglip'
        self._unfreeze(attention_probe)

    def _unfreeze(self, attention_probe):
        for param in self.backbone.parameters():
            param.requires_grad = False

        if attention_probe:
            if hasattr(self.backbone, 'attn_pool'):
                for param in self.backbone.attn_pool.parameters():
                    param.requires_grad = True
            elif hasattr(self.backbone, 'head'):
                for param in self.backbone.head.parameters():
                    param.requires_grad = True
            else:
                print('No pooling layer found')
                sys.exit(1)


    def forward(self, x):
        if self.backbone_type == 'siglip':
            hidden = self.backbone(pixel_values=x).pooler_output # [B, backbone_output_dim]
        else:
            hidden = self.backbone(x) # [B, backbone_output_dim]
        return self.linear(hidden) # [B, n_out_classes]

    def save(self, path: str, epoch: int, optimizer: Optimizer, scheduler: torch.optim.lr_scheduler._LRScheduler=None):
        """Saves a checkpoint with model, optimizer, AND scheduler states."""
        try:
            trainable_state_dict = {
                name: param for name, param in self.named_parameters() if param.requires_grad
            }
            if scheduler is None:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': trainable_state_dict,
                    'optimizer_state_dict': optimizer.state_dict(),
                }
            else:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': trainable_state_dict,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() 
                }
            torch.save(checkpoint, path)
            print(f"Successfully saved checkpoint to: {path}")
        except Exception as e:
            print(f"Error saving checkpoint to {path}: {e}")

    def load(self, path: str, optimizer: Optimizer = None, scheduler: torch.optim.lr_scheduler._LRScheduler = None, device: str = 'cuda'):
        """Loads model, optimizer, AND scheduler states from a checkpoint."""
        try:
            checkpoint = torch.load(path, map_location=device)
            self.load_state_dict(checkpoint['model_state_dict'], strict=False)
            
            if optimizer and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            if scheduler and 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                
            print(f"Successfully loaded probe weights from: {path}")
            return checkpoint.get('epoch', 0)
        except FileNotFoundError:
            print(f"Error: Weight file not found at '{path}'")
            return 0
        except Exception as e:
            print(f"An error occurred while loading the probe weights: {e}")
            return 0
        
    
    def save_head(self, path: str):
        """
        Saves ONLY the state_dict of the final linear layer (the head).
        This is intended for transferring the trained head to other models.
        """
        try:
            torch.save(self.linear.state_dict(), path)
            print(f"Successfully saved final head weights to: {path}")
        except Exception as e:
            print(f"Error saving final head to {path}: {e}")