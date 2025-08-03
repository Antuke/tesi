import sys

from transformers import AutoModel, AutoProcessor 
REPO_PATH = "C:/Users/antonio/Desktop/perception_models/"
sys.path.append(REPO_PATH)

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as transforms_pe
from core.vision_encoder.config import PE_VISION_CONFIG, PEConfig, fetch_pe_checkpoint
from dataclasses import asdict
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.optim import Optimizer
from torch.utils.data import random_split
from PIL import Image
from torch.optim.lr_scheduler import _LRScheduler


import matplotlib.pyplot as plt
import seaborn as sns

def _get_backbone_pe(ckpt, version):
    backbone_config = PE_VISION_CONFIG[version]
    transform = transforms_pe.get_image_transform_fix(image_size=backbone_config.image_size)
    v_cfg = asdict(backbone_config)
    backbone = pe.VisionTransformer(**v_cfg)
    backbone.load_ckpt(ckpt)
    return backbone, transform, backbone_config.output_dim

def get_backbone_pe(version):
    """
    Load PE ViT model, return model, transforms and size of output (dimension of embedding of last token)
    """
    backbone = pe.VisionTransformer.from_config(version, pretrained=True)
    backbone_config = PE_VISION_CONFIG[version]
    transform = transforms_pe.get_image_transform_fix(image_size=backbone_config.image_size)
    return backbone, transform, backbone_config.output_dim

def get_backbone_siglip2(model_name: str='google/siglip2-base-patch16-224'):
    """
    Load siglip2 ViT model, return model, transforms and size of output (dimension of embedding of last token)
    """
    print(f"Loading Hugging Face model: {model_name}")
    processor = AutoProcessor.from_pretrained(model_name)


    # Extract image processing configuration from the loaded processor
    image_processor_config = processor.image_processor
    image_size = image_processor_config.size['height']
    image_mean = image_processor_config.image_mean
    image_std = image_processor_config.image_std

    transform = transforms.Compose([
        transforms.Lambda(_convert_to_rgb),
        transforms.Resize((image_size, image_size), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_mean, std=image_std)
    ])
    
    # Load the model and return only the vision backbone
    model = AutoModel.from_pretrained(model_name)
    vision_model = model.vision_model
    print(f'Vision model = {vision_model}')
    return vision_model, transform, vision_model.config.hidden_size

def _convert_to_rgb(image: Image.Image) -> Image.Image:
    """Converts a PIL Image to RGB format."""
    return image.convert("RGB")


def get_backbone(version: str, ckpt : str=None):
    """
    Returns vision transformer backbone
    Args:
        version: Name of the backbone to use, PE-Core or Siglip
        ckpt: if different from null, loads backbone from .pt file specified, only for PE
    """
    if 'PE-Core-' in version:
        if ckpt is not None:
            return _get_backbone_pe(ckpt, version)
        else:
            return get_backbone_pe(version)
    else:
        return get_backbone_siglip2(version)

def log_to_disk(log_dir, message, mode):
    """
    Logs a message to a file on disk.

    Args:
        log_dir: Directory where the training_log.txt file is
        message: Message to save on the file
        mode: prefix of training_log.txt
    """
    # Ensure the target directory exists.
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, f'{mode}_training_log.txt')
    header = 'epoch,train_loss,val_loss,lr'

    # Check if the file needs a header. This is true if the file does not exist.
    write_header = not os.path.exists(log_path)

    with open(log_path, 'a') as f:
        if write_header:
            f.write(f'{header}\n')
        f.write(f'{message}\n')



def save_checkpoint(epoch: int, model: nn.Module, optimizer: Optimizer, scheduler: _LRScheduler, path: str):
    """
    Saves the model, optimizer, and scheduler state to a checkpoint file.
    
    Args:
        epoch (int): The current epoch number.
        model (nn.Module): The model to save.
        optimizer (Optimizer): The optimizer to save.
        scheduler (_LRScheduler): The learning rate scheduler to save.
        path (str): The path to save the checkpoint file to.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }
    path_with_ext = path + '_checkpoint.pt'
    torch.save(checkpoint, path_with_ext)
    print(f"Checkpoint saved to {path_with_ext}")