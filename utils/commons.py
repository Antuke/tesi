import sys
import os
from transformers import AutoModel, AutoProcessor 
from dotenv import load_dotenv
load_dotenv()
sys.path.append(os.getenv("REPO_PATH"))

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
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
import numpy as np

import matplotlib.pyplot as plt

def _get_backbone_pe(ckpt, version):
    backbone_config = PE_VISION_CONFIG[version]
    transform = transforms_pe.get_image_transform_fix(image_size=backbone_config.image_size)
    v_cfg = asdict(backbone_config)
    backbone = pe.VisionTransformer(**v_cfg)
    backbone.load_ckpt(ckpt)
    return backbone, transform, backbone_config.output_dim

def get_backbone_pe(version, print_info=False, apply_migration_flag=False):
    """
    Load PE ViT model, return model, transforms and size of output (dimension of embedding of last token)
    """
    print(f'Loading {version}...')
    backbone = pe.VisionTransformer.from_config(version, pretrained=True)
    backbone_config = PE_VISION_CONFIG[version]
    transform = transforms_pe.get_image_transform_fix(image_size=backbone_config.image_size)
    print("==============================")
    print(transform)
    print(f"applying migration = {apply_migration_flag}")
    print("==============================")
    if print_info:
        attnpool= backbone.attn_pool
        print(f'embed_dim={attnpool.embed_dim}\nnum_heads={attnpool.num_heads}')
        print(f'OUTPUT DIM = {backbone_config.output_dim}')

    def apply_migration(m):
        if isinstance(m, pe.SelfAttention):
            m.migrate_weights()

    if apply_migration_flag == True: # when testing/resuming no migration should be used
        print('[MIGRATION] Migrating weights for PEFT compatibiltyy')
        backbone.apply(apply_migration)

    return backbone, transform, backbone_config.output_dim


def get_backbone_dinov3(model_name: str="facebook/dinov3-vitb16-pretrain-lvd1689m", print_info=False):
    print(f"Loading Hugging Face model: {model_name}")
    processor = AutoProcessor.from_pretrained(model_name)
    if print_info:
        print(f'SIGLIP PROCESSOR:\n******************\n {processor}\n******************\n')

    # Extract image processing configuration from the loaded processor
    image_processor_config = processor
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
    vision_model = AutoModel.from_pretrained(model_name)

    if print_info:
        print(f'\nVISION CONFIGS:\n{vision_model.config}')
        print(f'\n\n\n{vision_model}')


    return vision_model, transform, vision_model.config.hidden_size


def get_backbone_siglip2(model_name: str='google/siglip2-base-patch16-224',print_info=False):
    """
    Load siglip2 ViT model, return model, transforms and size of output (dimension of embedding of last token)
    """
    print(f"Loading Hugging Face model: {model_name}")
    processor = AutoProcessor.from_pretrained(model_name)
    if print_info:
        print(f'SIGLIP PROCESSOR:\n******************\n {processor.image_processor}\n******************\n')

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

    if print_info:
        print(f'\nVISION CONFIGS:\n{vision_model.config}')
        print(f'\n\n***************MHAP\n{vision_model.head}')


    return vision_model, transform, vision_model.config.hidden_size

def _convert_to_rgb(image: Image.Image) -> Image.Image:
    """Converts a PIL Image to RGB format."""
    return image.convert("RGB")


def get_backbone(version: str, apply_migration : bool = False):
    """
    Returns vision transformer backbone
    Args:
        version: Name of the backbone to use, PE-Core or Siglip
        ckpt: if different from null, loads backbone from .pt file specified, only for PE
    """
    if 'PE-Core-' in version:
        return get_backbone_pe(version, False, apply_migration)
    elif 'siglip2' in version:
        print('[LOADING SIGLIP2]')
        return get_backbone_siglip2(version)
    elif 'dinov3' in version:
        return get_backbone_dinov3(version)
    
def log_to_disk(log_dir, message, mode, header = 'epoch,train_loss,val_loss,lr'):
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




def convert_labels(labels):
    """Converts so to compare with different age-group"""
    new_labels = []
    for label in labels:
        if label in ["0-2", "3-9"]:
            new_labels.append("0-9")
        elif label in ["10-19"]:
            new_labels.append("10-19")
        elif label in ["20-29", "30-39"]:
            new_labels.append("20-39")
        elif label in ["40-49", "50-59"]:
            new_labels.append("40-59")
        elif label in ["60-69", "70+"]:
            new_labels.append("60+")
        else:
            # Handle any unexpected labels
            new_labels.append(label)
    return new_labels


def hist(gate_Stats):
    """
    gate_Stats = {
        'gate_0': torch.tensor([124842.,  79691.,  42921.,  21525.,  40396., 217487., 155440.,  12710.]),
        'gate_1': torch.tensor([ 87632., 165969.,  66953., 165881.,  85621.,  15755., 102118.,   5083.]),
        'gate_2': torch.tensor([ 39121.,  25036., 160857.,  20101., 185666.,  13659.,   9452., 241120.])
    }
    """

    # --- Setup for plotting ---
    output_dir = 'moe_activation_charts'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    expert_labels = [f'Expert {i}' for i in range(8)]
    x_pos = np.arange(len(expert_labels))

    # --- Plot 1, 2, 3: Bar Chart for Each Individual Gate ---
    print("Generating individual gate activation charts...")
    for gate_name, data in gate_Stats.items():
        plt.figure(figsize=(12, 7))
        
        plt.bar(x_pos, data.numpy(), edgecolor='black', color='royalblue')
        
        plt.title(f'Activations for {gate_name}')
        plt.xlabel('Expert')
        plt.ylabel('Activation Value')
        plt.xticks(x_pos, expert_labels, rotation=45, ha="right")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # Save the individual chart to a file
        chart_path = os.path.join(output_dir, f'{gate_name}_activations.png')
        plt.savefig(chart_path)
        print(f"-> Saved chart to: {chart_path}")
        plt.close() # Close the figure to free up memory

    # --- Plot 4: Bar Chart of the Summed Activations ---
    print("\nGenerating summed activation chart...")

    # Calculate the element-wise sum of the tensors
    summed_activations = gate_Stats['gate_0'] + gate_Stats['gate_1'] + gate_Stats['gate_2']

    plt.figure(figsize=(12, 7))

    plt.bar(x_pos, summed_activations.numpy(), color='teal', edgecolor='black')

    plt.title('Total Activation per Expert (Sum of All Gates)')
    plt.xlabel('Expert')
    plt.ylabel('Total Activation Value')
    plt.xticks(x_pos, expert_labels, rotation=45, ha="right")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()

    # Save the summed chart to a file
    summed_chart_path = os.path.join(output_dir, 'summed_activations.png')
    plt.savefig(summed_chart_path)
    print(f"-> Saved summed chart to: {summed_chart_path}")
    plt.close()

    print("\nAll charts have been saved successfully.")
