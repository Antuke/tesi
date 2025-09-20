import os
import sys
import json
import csv
import random
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import numpy as np
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from torchvision import transforms


load_dotenv()
REPO_PATH = os.getenv("REPO_PATH")
if REPO_PATH:
    sys.path.append(REPO_PATH)

from utils.commons import *
from utils.datasets import get_split, resample, MTLDataset
from utils.dataset import *
from config.task_config import MTLConfig
from multitask.mtl_model import MTLModel
from config.task_config import Task




def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed) # Numpy module.
    random.seed(seed) # Python random module.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)

# --------------------------------------------------------- #
#                   Configurazione                          #
# --------------------------------------------------------- #
NAME = 'test'
DEVICE = 'cuda'

# Training hyperparams
RANK = 64
LEARNING_RATE= 5e-5
BACKBONE_LR_RATIO = 1
BATCH_SIZE = 100
NUM_WORKERS = 4


# MTL loss balancing method
USE_UW = False
USE_RUNNING_MEANS = True
EMA_ALPHA = 0.95


# Model choice
BACKBONE_NAME = 'PE-Core-L14-336' # PE-Core-S16-384 # PE-Core-B16-224 # PE-Core-L14-336 # PE-Core-T16-384
device = 'cuda'


# Datasets path
IMAGE_BASE_ROOT = "/user/asessa/dataset tesi/"
LABELS_ROOT = "/user/asessa/dataset tesi/LABELS"
DATASET_NAMES = ["FairFace",  "RAF-DB", "Lagenda", "CelebA_HQ"]


# Scheduler Configuration
SCHEDULER_PATIENCE = 5
SCHEDULER_FACTOR = 0.1

# Early Stopping Configuration
EPOCHS=100
USE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 10

# Tasks configuration
NUM_TASKS = 3
TASKS=[
    Task(name='Age', class_labels=["0-2","3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"], criterion=torch.nn.CrossEntropyLoss, weight=1.0, use_weighted_loss=False),
    Task(name='Gender', class_labels=["Male", "Female"], criterion=torch.nn.CrossEntropyLoss, weight=1.0, use_weighted_loss=False),
    Task(name='Emotion', class_labels=["Surprise", "Fear", "Disgust", "Happy", "Sad", "Angry", "Neutral"], criterion=torch.nn.CrossEntropyLoss, weight=1.0, use_weighted_loss=False)
]
TASK_NAMES = [task.name for task in TASKS]
AGE_IDX, GENDER_IDX, EMOTION_IDX = 0, 1, 2


# ----------------------------------------------------------------- #
#                   Output & Logging Setup                          #
# ------------------------------------------------------------------#
run_name = f"run_{NAME}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
output_dir = Path("./training_outputs") / run_name
ckpt_dir = output_dir / "ckpt"
cm_dir = output_dir / "confusion_matrices"
output_dir.mkdir(parents=True, exist_ok=True)
ckpt_dir.mkdir(exist_ok=True)
cm_dir.mkdir(exist_ok=True)
for task in TASKS:
    (cm_dir / task.name).mkdir(exist_ok=True)

print(f"Outputs will be saved to: {output_dir}")


config = {
    "run_name": run_name,
    "backbone": BACKBONE_NAME,
    "rank": RANK,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "epochs": EPOCHS,
    "use_uncertainty_weighting": USE_UW,
    "use_running_mean": USE_RUNNING_MEANS,
    "scheduler": {
        "type": "ReduceLROnPlateau",
        "patience": SCHEDULER_PATIENCE,
        "factor": SCHEDULER_FACTOR
    },
    "early_stopping": {
        "enabled": USE_EARLY_STOPPING,
        "patience": EARLY_STOPPING_PATIENCE
    },
    "tasks": [
        {
            "name": task.name, "class_labels": task.class_labels, "criterion": task.criterion.__name__,
            "weight": task.weight, "use_weighted_loss": task.use_weighted_loss,
        } for task in TASKS
    ],
    "train_dataset": DATASET_NAMES
}
with open(output_dir / "config.json", "w") as f:
    json.dump(config, f, indent=4)
print("Configuration saved to config.json")


# ---------------------------------------------------------------------------- #
#                   Model setup e optimizer                                    #
# -----------------------------------------------------------------------------#
backbone, transform, hidden_size = get_backbone(BACKBONE_NAME, apply_migration = True)



model = MTLModel(backbone, TASKS, rank=RANK).to(device)
model = torch.compile(model)
# Freeze all parameters
for name, param in model.named_parameters():
    param.requires_grad = False


# Unfreeze lora matrices, task-specific attention pooling heads, and prediction layers
for name, param in model.named_parameters():
    if "lora_" in name or ("attn_pool" in name and "backbone" not in name) or "prediction_layers" in name:
        param.requires_grad = True
        print(f'ADDING TO OPTIMIZERS = {name}')


head_lr_params = [p for n, p in model.named_parameters() if p.requires_grad and ("attn_pool" in n or "prediction_layers" in n)]
backbone_lr_params = [p for n, p in model.named_parameters() if p.requires_grad and not ("attn_pool" in n or "prediction_layers" in n)]

if USE_UW:
    # This logic is mutually exclusive with USE_RUNNING_MEANS in this setup
    assert not USE_RUNNING_MEANS, "Cannot use USE_UW and USE_RUNNING_MEANS at the same time."
    model.log_vars.requires_grad=True
    head_lr_params.append(model.log_vars)

optimizer_grouped_parameters = [
    {'params': head_lr_params, 'lr': LEARNING_RATE},
    {'params': backbone_lr_params, 'lr': LEARNING_RATE * BACKBONE_LR_RATIO}
]

print_trainable_params(model)


optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
scaler = torch.amp.GradScaler('cuda')
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE)


model.enable_gradient_checkpointing() # To be able to train large model and not run out of VRAM (can be removed if training Base ViT or lower)



# ---------------------------------------------------------------#
#                   Datasets & Loaders                           #
# ---------------------------------------------------------------#
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomRotation(degrees=10),
    transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
    *transform.transforms,
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3))
])

train_dataset = TaskBalanceDataset(
    dataset_names=DATASET_NAMES,
    transform=train_transforms,
    split="train",
    datasets_root=LABELS_ROOT,
    image_base_root=IMAGE_BASE_ROOT,
    verbose=True,
    balance_task={EMOTION_IDX: 0.33},
    augment_duplicate=None
)

age_weights = train_dataset.get_class_weights(AGE_IDX, 'default')
gender_weights = train_dataset.get_class_weights(GENDER_IDX, 'default')
emotion_weights = train_dataset.get_class_weights(EMOTION_IDX, 'default')
class_weights = [age_weights, gender_weights, emotion_weights]

train_sampler, sample_weights = build_weighted_sampler(
    dataset=train_dataset,
    class_weights_per_task=class_weights,
    device=DEVICE,
    combine='mean'
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    sampler=train_sampler,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

val_dataset = MultiDataset(
    dataset_names=DATASET_NAMES,
    transform=transform,
    split="val",
    datasets_root=LABELS_ROOT,
    image_base_root=IMAGE_BASE_ROOT,
    verbose=True
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)



# --------------------------------------------------#
#                   Loss functions                  #
# --------------------------------------------------#
# The balanced sampling strategies allows us to use standard CrossEntropy as a loss function
criterions = {}
for task in TASKS:
    params = {'ignore_index': -100}
    criterions[task.name] = task.criterion(**params)


# ----------------------------------------------------#
#                   CSV logger setup                  #
# ----------------------------------------------------#
log_path = output_dir / "log.csv"
log_header = ['epoch', 'learning_rate', 'avg_train_loss', 'avg_val_loss']
log_header.extend([f'{task.name}_train_loss' for task in TASKS])
log_header.extend([f'{task.name}_val_loss' for task in TASKS])
log_header.extend([f'{task.name}_val_acc' for task in TASKS])
if USE_UW:
    log_header.extend([f'{task.name}_variance' for task in TASKS])
if USE_RUNNING_MEANS:
    log_header.extend([f'{task.name}_weight' for task in TASKS])

with open(log_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(log_header)
print("CSV log file created at log.csv")

# --------------------------------------------------------------#
#                   Training & Validation loop                  #
# --------------------------------------------------------------#
best_val_loss = float('inf')
best_age_acc = 0.0
best_emotion_acc = 0.0
best_avg_acc = 0.0
epochs_no_improve = 0

if USE_RUNNING_MEANS:
    running_means = RunningMeans(task_names=TASK_NAMES, alpha=EMA_ALPHA)

for epoch in range(EPOCHS):
    model.train()
    running_train_loss = 0.0
    task_train_losses = {task.name: 0.0 for task in TASKS}
    task_train_counts = {task.name: 0 for task in TASKS}
    
    # --- DYNAMIC WEIGHT CALCULATION  ---
    task_weights = {}
    if USE_RUNNING_MEANS:
        raw_weights = []
        for idx, task in enumerate(TASKS):
            # Get the running mean from the previous epoch
            running_mean_val = running_means.get_by_index(idx)
            
            # if EMA is not initialized, use the static weight from config 
            if running_mean_val is None:
                # Use the inverse of the static weight as a starting point
                raw_weights.append(1.0 / task.weight)
            else:
                # The core logic: weight is inverse to the running mean loss
                raw_weights.append(1.0 / max(running_mean_val, 1e-8))
        
        # Normalize the raw weights by their average to keep the scale consistent
        avg_raw_weight = sum(raw_weights) / len(raw_weights)
        
        # Final weights for this epoch
        final_weights = [w / avg_raw_weight for w in raw_weights]
        
        for i, task in enumerate(TASKS):
            task_weights[task.name] = final_weights[i]
        
        print(f"Epoch {epoch+1} Task Weights (Running Means): {task_weights}")
    else:
        # Default: Use fixed weights from the config
        for task in TASKS:
            task_weights[task.name] = task.weight
        print(f"Epoch {epoch+1} Task Weights (Static): {task_weights}")


    progress_bar_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
    for i, (images, labels) in enumerate(progress_bar_train):
        images, labels = images.to(device), labels.to(device)
        gt_labels = {task.name: labels[:, i] for i, task in enumerate(TASKS)}

        optimizer.zero_grad()
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
            outputs = model(images)
            total_loss = 0
            
            # ---  Calculate individual task losses and update running means ---
            current_batch_task_losses = {}
            for idx, task in enumerate(TASKS):
                valid_indices = gt_labels[task.name] != -100
                if torch.any(valid_indices):
                    task_loss = criterions[task.name](outputs[task.name][valid_indices], gt_labels[task.name][valid_indices])
                    current_batch_task_losses[task.name] = task_loss
                    
                    # Update epoch-level stats for logging
                    task_train_losses[task.name] += task_loss.item() * valid_indices.sum().item()
                    task_train_counts[task.name] += valid_indices.sum().item()

                    # Update running mean for the *next* epoch's weight calculation
                    if USE_RUNNING_MEANS:
                        running_means.update_by_idx(task_loss.item(), idx)

            # ---  Combine losses using the epoch-level weights ---
            for task_name, task_loss in current_batch_task_losses.items():
                total_loss += task_weights[task_name] * task_loss
        
        # --- Backpropagation ---
        if isinstance(total_loss, torch.Tensor) and total_loss != 0:
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_train_loss += total_loss.item()
            
        progress_bar_train.set_postfix(loss=f"{running_train_loss / (progress_bar_train.n + 1):.4f}")

    # --- Validation Phase ---
    model.eval()
    task_val_losses = {task.name: 0.0 for task in TASKS}
    task_val_counts = {task.name: 0 for task in TASKS}
    all_preds = {task.name: [] for task in TASKS}
    all_labels = {task.name: [] for task in TASKS}

    with torch.no_grad():
        progress_bar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]")
        for images, labels in progress_bar_val:
            images, labels = images.to(device), labels.to(device)
            gt_labels = {task.name: labels[:, i] for i, task in enumerate(TASKS)}

            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                outputs = model(images)
                for task in TASKS:
                    task_name = task.name
                    valid_indices = gt_labels[task_name] != -100

                    if torch.any(valid_indices):
                        task_loss = criterions[task_name](outputs[task_name][valid_indices], gt_labels[task_name][valid_indices])
                        
                        task_val_losses[task_name] += task_loss.item() * valid_indices.sum().item()
                        task_val_counts[task_name] += valid_indices.sum().item()
                        
                        _, predicted = torch.max(outputs[task_name].data, 1)
                        all_preds[task_name].extend(predicted[valid_indices].cpu().numpy())
                        all_labels[task_name].extend(gt_labels[task_name][valid_indices].cpu().numpy())

    # --- Epoch Summary & Logging  ---
    avg_train_loss = running_train_loss / len(train_loader)
    avg_task_val_losses = {name: (task_val_losses[name] / task_val_counts[name]) if task_val_counts[name] > 0 else 0.0 for name in TASK_NAMES}
    avg_task_train_losses = {name: (task_train_losses[name] / task_train_counts[name]) if task_train_counts[name] > 0 else 0.0 for name in TASK_NAMES}

    # For validation loss, we always use the static weights for a consistent comparison metric
    static_val_weights = {task.name: task.weight for task in TASKS}
    avg_val_loss = sum(avg_task_val_losses[name] * static_val_weights[name] for name in TASK_NAMES)

    scheduler.step(avg_val_loss)
    current_lr = optimizer.param_groups[0]['lr']

    
    print(f"\n--- Epoch {epoch+1}/{EPOCHS} Summary ---")
    print(f"  Avg Train Loss (Combined): {avg_train_loss:.4f}")
    print(f"  Avg Val Loss (Unweighted Avg): {avg_val_loss:.4f}")
    print(f"  Current LR: {current_lr:g}")

    accuracies = {task_name: accuracy_score(all_labels[task_name], all_preds[task_name]) if all_labels[task_name] else 0.0 for task_name in TASK_NAMES}
    avg_accuracy = np.mean(list(accuracies.values()))

    print("  Validation Metrics:")
    for task_name in TASK_NAMES:
        print(f"    - {task_name}: Acc: {accuracies[task_name]:.4f}, Loss: {avg_task_val_losses[task_name]:.4f}")
    
    message = f"""*Epoch {epoch+1} Summary*\n*Average Validation Loss:* `{avg_val_loss:.4f}`\n*Validation Accuracies:*\n"""
    for task_name in TASK_NAMES:
        message += f"- *{task_name}:* `{accuracies[task_name]:.4f}`\n"
    send_telegram_message(message)
    
    # --- Confusion Matrices ---
    for task in TASKS:
        task_name = task.name
        if all_labels[task_name]:
            cm = confusion_matrix(all_labels[task_name], all_preds[task_name])
            cm_normalized = confusion_matrix(all_labels[task_name], all_preds[task_name], normalize='true')

            # Plot and save non-normalized confusion matrix
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=task.class_labels, yticklabels=task.class_labels)
            plt.title(f'{task_name} Confusion Matrix (Epoch {epoch+1})')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.savefig(cm_dir / task_name / f'cm_epoch_{epoch+1}.png')
            plt.close()

            # Plot and save normalized confusion matrix
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=task.class_labels, yticklabels=task.class_labels)
            plt.title(f'{task_name} Normalized Confusion Matrix (Epoch {epoch+1})')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.savefig(cm_dir / task_name / f'cm_normalized_epoch_{epoch+1}.png')
            plt.close()


    log_row = [
        epoch + 1, round(current_lr, 8), round(avg_train_loss, 5), round(avg_val_loss, 5),
        *[round(avg_task_train_losses.get(task.name, 0.0), 5) for task in TASKS],
        *[round(avg_task_val_losses.get(task.name, 0.0), 5) for task in TASKS],
        *[round(accuracies.get(task.name, 0.0), 5) for task in TASKS],
    ]
    if USE_UW:
        variances = [torch.exp(log_var).item() for log_var in model.log_vars]
        log_row.extend([round(v, 5) for v in variances])

    if USE_RUNNING_MEANS:
        # Log the weights that were used for this epoch
        log_row.extend([round(task_weights.get(task.name, 0.0), 5) for task in TASKS])


    with open(log_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(log_row)

    # --- Early Stopping & Model Checkpointing ---
    if avg_val_loss < best_val_loss:
        print(f"Validation loss decreased ({best_val_loss:.4f} --> {avg_val_loss:.4f}). Saving model...")
        best_val_loss = avg_val_loss
        save_path = ckpt_dir / f'best_model_{epoch+1}.pt'
        model.save_model(str(save_path))
        print(f"Model saved to {save_path}")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        print(f"Validation loss did not improve for {epochs_no_improve} epoch(s).")
    
    # --- Save model based on best age accuracy ---
    if accuracies['Age'] > best_age_acc:
        print(f"Age accuracy improved ({best_age_acc:.4f} --> {accuracies['Age']:.4f}). Saving model...")
        best_age_acc = accuracies['Age']
        save_path = ckpt_dir / f'best_model_{epoch+1}.pt'
        model.save_model(str(save_path))
        print(f"Model saved to {save_path}")

    # --- Save model based on best avg accuracy ---
    if avg_accuracy > best_avg_acc:
        print(f"Average accuracy improved ({best_avg_acc:.4f} --> {avg_accuracy:.4f}). Saving model...")
        best_avg_acc = avg_accuracy
        save_path = ckpt_dir / f'best_model_{epoch+1}.pt'
        model.save_model(str(save_path))
        print(f"Model saved to {save_path}")
    
    # --- Save model based on best emotion accuracy ---
    if accuracies['Emotion'] > best_emotion_acc:
        print(f"Emotion accuracy improved ({best_emotion_acc:.4f} --> {accuracies['Emotion']:.4f}). Saving model...")
        best_emotion_acc = accuracies['Emotion']
        save_path = ckpt_dir / f'best_model_{epoch+1}.pt'
        model.save_model(str(save_path))
        print(f"Model saved to {save_path}")
    

    if USE_EARLY_STOPPING and epochs_no_improve >= EARLY_STOPPING_PATIENCE:
        print(f"\n--- Early stopping triggered after {EARLY_STOPPING_PATIENCE} epochs with no improvement. ---")
        print(f"Best model with val_loss {best_val_loss:.4f} was saved.")
        break

    print("-" * 50)