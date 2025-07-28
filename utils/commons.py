import sys

from transformers import AutoModel, AutoProcessor 
REPO_PATH = "C:/Users/antonio/Desktop/perception_models/"
sys.path.append(REPO_PATH)

from utils.dataset import AgeDataset
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as transforms_pe
from core.vision_encoder.config import PE_VISION_CONFIG, PEConfig, fetch_pe_checkpoint
from utils.dataset import AgeDataset
from dataclasses import asdict
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.optim import Optimizer
from torch.utils.data import random_split
from PIL import Image
from torch.optim.lr_scheduler import _LRScheduler


import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(cm, class_names, ylabel="True Age Group", xlabel="Predicted Age Group"):
    """
    Generates and displays a heatmap for a given confusion matrix.
    
    Args:
       cm (array): A confusion matrix.
       class_names (list): A list of category names for the axes.
    """
    plt.figure(figsize=(12, 10))
    heatmap = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                          xticklabels=class_names, yticklabels=class_names)
    
    heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0) 
    heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45, ha='right')
    
    plt.title("Confusion Matrix", fontsize=16)
    plt.ylabel(ylabel, fontsize=12)
    plt.xlabel(xlabel, fontsize=12)
    plt.show()

def get_backbone_pe(ckpt, version):
    backbone_config = PE_VISION_CONFIG[version]
    transform = transforms_pe.get_image_transform_fix(image_size=backbone_config.image_size)
    v_cfg = asdict(backbone_config)
    backbone = pe.VisionTransformer(**v_cfg)
    backbone.load_ckpt(ckpt)
    return backbone, transform, backbone_config.output_dim


def get_backbone_siglip2(model_name: str='google/siglip2-base-patch16-224'):
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
    
    return vision_model, transform, vision_model.config.hidden_size

def _convert_to_rgb(image: Image.Image) -> Image.Image:
    """Converts a PIL Image to RGB format."""
    return image.convert("RGB")


def get_backbone(version: str, ckpt : str=None):
    if 'PE-Core-' in version:
        return get_backbone_pe(ckpt, version)
    else:
        return get_backbone_siglip2(version)

def log_to_disk(log_dir, message, mode):
    """
    Logs a message to a file on disk.
    """
    # Ensure the target directory exists.
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, f'{mode}_training_log.txt')
    header = 'epoch,train_loss,val_loss'

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
    # Ensure the directory exists
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