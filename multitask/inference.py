import torch 
import torch.nn as nn 
from multitask.probe import MultiTaskProbe
from utils.commons import get_backbone
from config.task_config import Task
from PIL import Image
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from typing import List, Dict, Any, Optional
import numpy as np
import matplotlib.colors as colors

def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """
    De-normalizes a PyTorch tensor and converts it to a displayable NumPy image.

    Args:
        tensor: The input tensor, expected shape [1, 3, H, W] or [3, H, W].

    Returns:
        A NumPy array of shape [H, W, 3] with pixel values in [0, 1].
    """
    # Standard ImageNet normalization constants used by many models like SigLIP
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    # Squeeze the batch dimension if it exists
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)

    # Move tensor to CPU and clone it to avoid modifying the original
    tensor = tensor.cpu().clone()

    # De-normalize: tensor = tensor * std + mean
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
        
    # Rearrange dimensions from [C, H, W] to [H, W, C]
    image_np = tensor.numpy().transpose(1, 2, 0)
    
    # Clip values to be between 0 and 1
    image_np = np.clip(image_np, 0, 1)
    
    return image_np


versions = [
    'PE-Core-B16-224',
    'google/Siglip2-base-patch16-224'
]

tasks=[
        Task(name='Age', class_labels=["0-2","3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"], criterion=nn.CrossEntropyLoss, weight=1.0, use_weighted_loss=True),
        Task(name='Gender', class_labels=["Male", "Female"], criterion=nn.CrossEntropyLoss, weight=1.0),
        Task(name='Emotion', class_labels=["Surprise", "Fear", "Disgust", "Happy", "Sad", "Angry", "Neutral"], criterion=nn.CrossEntropyLoss, weight=1.0, use_weighted_loss=True)
]


import matplotlib.colors as colors

def visualize_attention_maps(
    model_output: Dict[str, Any],
    transformed_image_tensor: torch.Tensor,
    tasks: List[Task],
    save_path: Optional[str] = None,
    # New arguments for better visualization
    scaling_method: str = 'log', # Options: 'log', 'clip', 'linear',
    pe: bool = True
):
    """
    Visualizes attention maps with improved scaling to reveal more detail.
    """
    print(model_output)
    if 'attn_weights' not in model_output: return
    display_image = denormalize_image(transformed_image_tensor)
    img_height, img_width, _ = display_image.shape
    attn_weights = model_output['attn_weights'].cpu()
    
    if pe:
        attn_weights = attn_weights[:, :, 1:]
    num_probes, num_patches = attn_weights.shape[1], attn_weights.shape[2]
    grid_size = int(np.sqrt(num_patches))
    
    fig, axes = plt.subplots(1, len(tasks), figsize=(len(tasks) * 5, 5))
    if len(tasks) == 1: axes = [axes]

    print(f"\n--- Generating Attention Maps (Scaling: {scaling_method}) ---")

    for i, task in enumerate(tasks):
        task_attn_map = attn_weights[0, i, :]
        attn_grid = task_attn_map.detach().numpy().reshape(grid_size, grid_size)
        

        norm = None
        if scaling_method == 'log':
            # Use LogNorm. Add a small epsilon to avoid log(0).
            attn_grid_scaled = attn_grid + 1e-6
            norm = colors.LogNorm(vmin=attn_grid_scaled.min(), vmax=attn_grid_scaled.max())
        elif scaling_method == 'clip':
            # Clip the 99th percentile. This ignores extreme outliers.
            vmax = np.percentile(attn_grid, 99)
            norm = colors.Normalize(vmin=attn_grid.min(), vmax=vmax)
        else: # 'linear'
            attn_grid_scaled = attn_grid
            norm = colors.Normalize(vmin=attn_grid.min(), vmax=attn_grid.max())

        resized_attn_grid = resize(attn_grid, (img_height, img_width), order=3, mode='constant')

        ax = axes[i]
        ax.imshow(display_image)
        # Pass the normalization object to imshow
        im = ax.imshow(resized_attn_grid, cmap='jet', alpha=0.5, norm=norm)
        ax.set_title(f"Attention for Task: {task.name}")
        ax.axis('off')
        
    # Add a colorbar to understand the scale
    fig.colorbar(im, ax=axes.ravel().tolist(), orientation='horizontal', fraction=0.05, pad=0.05)
    plt.tight_layout()

    print(f"ATTENTION MAP SAVED TO: {save_path}")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    plt.close(fig)

def parse_logits(logits_dict: Dict[str, Any], tasks: List[Task]):
    """
    Parses the model's output dictionary to print classification results.

    Args:
        logits_dict: The output dictionary from the model.
        tasks: A list of Task objects defining the classification heads.
    """
    print("--- Classification Results ---")
    
    # Extract the list of logit tensors
    list_of_logits = logits_dict['logits']

    if len(list_of_logits) != len(tasks):
        print(f"Warning: Mismatch between number of logit outputs ({len(list_of_logits)}) and tasks ({len(tasks)}).")
        return

    # Iterate over each task and its corresponding logit tensor
    for task, logit_tensor in zip(tasks, list_of_logits):
        # Apply softmax to convert logits to probabilities
        probabilities = torch.nn.functional.softmax(logit_tensor, dim=1)
        
        # Get the index of the highest probability
        predicted_index = torch.argmax(probabilities, dim=1).item()
        
        # Get the corresponding class label
        predicted_label = task.class_labels[predicted_index]
        
        # Get the confidence score of the prediction
        confidence = probabilities[0, predicted_index].item()
        probs_str = [f"{p:.2f}" for p in probabilities.tolist()[0]]
        print(f"Task: {task.name:<10} | Prediction: {predicted_label:<15} | Confidence: {confidence:.2%}    | {probs_str}")

ckpts = ['/user/asessa/tesi/multitask/outputs_lora_default/ckpt/mtl_PE-Core-B16-224_ul6_30.pt',
        '/user/asessa/tesi/multitask/outputs_siglip_no_pt/ckpt/mtl_Siglip2_ul6_60.pt']

# --- Main Execution ---
if __name__ == '__main__':
    CHOSEN = 0
    backbone, transform, hidden_size = get_backbone(versions[CHOSEN])
    name = 'pe' if CHOSEN == 0 else 'siglip'
    task_dims = {task.name.lower(): task.num_classes for task in tasks}

    model = MultiTaskProbe(
            backbone=backbone,
            backbone_output_dim=hidden_size,
            tasks=task_dims,
            use_moe=False,
            use_k_probes=False,
            uncertainty_weighting_for_bal = 1,
            use_lora= True,
            deeper_classification_heads = False
    ).to('cuda')

    # Load your trained model checkpoint
    model.load(ckpts[CHOSEN])
    model.eval()
    model.log_probe_similarity()
    # Load the original image for visualization (keep it in PIL format)
    image_path = "/user/asessa/dataset tesi/datasets_with_standard_labels/FairFace/test/imagescropped/00086768.jpg"
    image_path = "/user/asessa/tesi/multitask/assets/sad_little_girl.jpg"
    original_pil_image = Image.open(image_path).convert("RGB")

    # Transform the image for the model
    transformed_img = transform(original_pil_image).unsqueeze(0).to('cuda')

    # Run inference and get the full output dictionary
    with torch.no_grad():
        model_output = model(transformed_img, return_attn_weights=True)

    #print('GATE ACTIVATIONS:')
    #print(model_output["stats"])
    # --- Step 1: Parse and print the classification results ---
    parse_logits(model_output, tasks)
    
    # --- Step 2: Visualize the attention maps ---
    visualize_attention_maps(
        model_output=model_output,
        transformed_image_tensor=transformed_img, # <-- Pass the tensor here
        tasks=tasks,
        save_path=f"./girl_{name}.jpg",
        pe = True if CHOSEN == 0 else False
    )