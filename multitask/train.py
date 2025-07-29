import argparse
import sys
import os
from dotenv import load_dotenv
load_dotenv()
sys.path.append(os.getenv("REPO_PATH"))

import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Subset
import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as transforms_pe
from core.vision_encoder.config import PE_VISION_CONFIG
from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer
from utils.dataset import *
from utils.commons import *
from multitask_probe import MultiTaskProbe
from sklearn.metrics import confusion_matrix
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR
from tqdm import tqdm 
import argparse


####---------- CONFIGS ----------####

QUICK = True
SAVE_FOLDER = "./outputs"
STATIC_TASK_WEIGHT = [1,1,1]
age_labels_name = ["0-2","0-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "more than 70"]
gender_labels_name = ["Male", "Female"]
emotion_labels_name = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
device = 'cuda'
TASKS = ['Gender classification','Age Classification','Emotion Classification']

def validate(loader: DataLoader, model: nn.Module, criterions: list):
    """
    Validates the multi-task model and return confusion matrices
    """
    model.eval()
    running_loss = 0.0
    task_losses = [0.0] * len(criterions)
    num_tasks = len(criterions)
    all_gender_preds, all_gender_true = [], []
    all_age_preds, all_age_true = [], []
    all_emotion_preds, all_emotion_true = [], []

    with torch.no_grad():
        loader_tqdm = tqdm(loader, desc="Validating")
        for batch_idx, (images, labels) in enumerate(loader_tqdm):
            inputs = images.to(device)
            labels = labels.to(device)
            task_outputs = model(inputs)
            current_batch_task_losses = [0.0] * num_tasks
            batch_total_loss = 0.0

            for i, criterion in enumerate(criterions):
                current_output = task_outputs[i]
                current_target = labels[:, i]

                # --- Loss Calculation ---
                if isinstance(criterion, nn.BCEWithLogitsLoss):
                    current_target = current_target.float().unsqueeze(1)
                    task_loss = criterion(current_output, current_target)
                else:
                    task_loss = criterion(current_output, current_target)

                
                task_losses[i] += task_loss.item()
                batch_total_loss += task_loss * STATIC_TASK_WEIGHT[i]
                current_batch_task_losses[i] = task_loss.item()
                valid_mask = (current_target != -100)
                if not valid_mask.any():
                    continue

                # Gender (Task 0)
                if i == 0:
                    preds = (torch.sigmoid(current_output[valid_mask]) > 0.5).int()
                    all_gender_preds.extend(preds.cpu().numpy())
                    all_gender_true.extend(current_target[valid_mask].cpu().numpy())
                # Age (Task 1)
                elif i == 1:
                    preds = current_output[valid_mask].argmax(dim=1)
                    all_age_preds.extend(preds.cpu().numpy())
                    all_age_true.extend(current_target[valid_mask].cpu().numpy())
                # Emotion (Task 2)
                elif i == 2:
                    preds = current_output[valid_mask].argmax(dim=1)
                    all_emotion_preds.extend(preds.cpu().numpy())
                    all_emotion_true.extend(current_target[valid_mask].cpu().numpy())

            running_loss += batch_total_loss.item()
            
            task_loss_str = ", ".join([f"T{i}: {l:.4f}" for i, l in enumerate(current_batch_task_losses)])
            avg_running_loss = running_loss / (batch_idx + 1)
            loader_tqdm.set_description(
                f"Validating - Avg Loss: {avg_running_loss:.4f} | Batch Task Losses: [{task_loss_str}]"
            )
    # Calculate final metrics
    avg_loss = running_loss / len(loader)
    avg_task_losses = [l / len(loader) for l in task_losses]
    
    # Create confusion matrices
    cm_gender = confusion_matrix(all_gender_true, all_gender_preds, labels=range(len(gender_labels_name)))
    cm_age = confusion_matrix(all_age_true, all_age_preds, labels=range(len(age_labels_name)))
    cm_emotion = confusion_matrix(all_emotion_true, all_emotion_preds, labels=range(len(emotion_labels_name)))

    return avg_loss, avg_task_losses, (cm_gender, cm_age, cm_emotion)

def train_epoch(loader, model, criterions, optimizer):
    model.train() 
    running_loss = 0.0

    num_tasks = len(criterions)
    task_losses = [0.0] * num_tasks
    samples_per_task = [0] * num_tasks
    
    loader_tqdm = tqdm(loader, desc="Training")

    for batch_idx, (images, labels) in enumerate(loader_tqdm):
        inputs = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        task_outputs = model(inputs) 

        loss = 0.0
        current_batch_task_losses = [0.0] * num_tasks

        for i, criterion in enumerate(criterions):
            current_output = task_outputs[i]
            current_target = labels[:, i]

            if (current_target == -100).all():
                continue

            samples_per_task[i] += (current_target != -100).sum().item()
            if isinstance(criterion, nn.BCEWithLogitsLoss):
                current_target = current_target.float().unsqueeze(1)

            task_loss = criterion(current_output, current_target)
            
            
            current_batch_task_losses[i] = task_loss.item()
            task_losses[i] += task_loss.item()

            loss += task_loss * STATIC_TASK_WEIGHT[i]

        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() 

        avg_running_loss = running_loss / (batch_idx + 1)
        
        task_loss_str = ", ".join([f"T{i}: {l:.4f}" for i, l in enumerate(current_batch_task_losses)])
        
        loader_tqdm.set_description(
            f"Training - Avg Loss: {avg_running_loss:.4f} | Batch Task Losses: [{task_loss_str}]"
        )

    avg_epoch_loss = running_loss / len(loader)
    avg_task_losses = [
        total_loss / (num_samples if num_samples > 0 else 1)
        for total_loss, num_samples in zip(task_losses, samples_per_task)
    ]

    return avg_epoch_loss, avg_task_losses



def train(
    version: str,
    epochs: int,
    dataset_root_dir: str,
    csv_path: str,
    num_layers_to_unfreeze: int = 0,
    ckpt_path: str = None,
    batch_size: int = 8,
    learning_rate: float = 0.001,
    resume_from_ckpt:str = None
    ):
    
    backbone, transform, hidden_size = get_backbone(version, ckpt_path)
    dataset = MergedDataset(dataset_root_dir,
                            csv_file=csv_path,
                            transform=transform)

    if QUICK:
        dataset = Subset(dataset,random.sample(range(len(dataset)), 5000))
    
        train_loader, val_loader, test_loader = get_loaders_broken_(dataset, 
                                                            torch.Generator(), 
                                                          batch_size=batch_size)
        
    train_loader, val_loader, test_loader = get_loaders_broken_(dataset, 
                                                            torch.Generator(), 
                                                            batch_size=batch_size)
    mtl_probe = MultiTaskProbe(backbone=backbone,
                               backbone_output_dim=hidden_size,
                               num_layers_to_unfreeze=num_layers_to_unfreeze).to(device)
    
    # three classification task, cross-entropy and binary cross-entropy
    # Pytorch implementation of CrossEntropyLoss already implements masked loss
    criterions = [
        nn.BCEWithLogitsLoss(weight=torch.tensor(GENDER_INVERSE_FREQ).to(device)),  # Each sample has the gender label
        nn.CrossEntropyLoss(weight=torch.tensor(AGE_INVERSE_FREQ).to(device),       ignore_index=-100),
        nn.CrossEntropyLoss(weight=torch.tensor(EMOTION_INVERSE_FREQ).to(device),   ignore_index=-100)
    ]

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, mtl_probe.parameters()), 
        lr=learning_rate
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    start_epoch = 0
    if resume_from_ckpt:
        if os.path.exists(resume_from_ckpt):
            start_epoch = mtl_probe.load(resume_from_ckpt, optimizer, scheduler)
        else:
            print(f"Checkpoint file not found at {resume_from_ckpt}, starting from scratch.")
    
    for i in range(start_epoch, epochs):
        print(f"\nEpoch {i+1}/{epochs}")

        train_loss,avg_task_losses = train_epoch(
            loader=train_loader, 
            model=mtl_probe, 
            criterions=criterions, 
            optimizer=optimizer
        )

        val_loss, avg_task_losses_val, _ = validate(val_loader,
                                                     model=mtl_probe,
                                                     criterions=criterions)
        

        print(f'Train Loss = {train_loss:.4f}\nTask losses = {[f'{x:.4f}' for x in avg_task_losses]}')
        log_to_disk(SAVE_FOLDER, f'{i+1},{train_loss:.5f},{",".join([f"{x:.4f}" for x in avg_task_losses])},{val_loss:.5f},{",".join([f"{x:.4f}" for x in avg_task_losses_val])}', 'MTL')
        
        mtl_probe.save(
            path=SAVE_FOLDER,
            epoch=i+1,
            optimizer=optimizer,
            scheduler=scheduler
        )

    print(f"\n--- Final Testing  ---")
    test_loss, avg_task_losses_test, cms = validate(val_loader,
                                                     model=mtl_probe,
                                                     criterions=criterions)
    print(f"Final Test Loss: {val_loss:.5f}\nTask losses = {[f'{x:.4f}' for x in avg_task_losses_test]}")
    for i, cm in enumerate(cms):
        print(f'\n\n{TASKS[i]} Confusion Matrix:\n{cm}')


    return test_loss, avg_task_losses_test, cms




def main():


    parser = argparse.ArgumentParser(description="Train and validate multi-task probe.")

    parser.add_argument('--version', type=str,
                       default='google/Siglip2-base-patch16-224',
                       help='Backbone model version.')
    parser.add_argument('--ckpt_path', type=str,
                       default='../ckpt/PE-Core-B16-224.pt',
                       help='Path to the backbone checkpoint.')
    parser.add_argument('--resume_from_ckpt', type=str, default=None,
                        help='Path to a checkpoint to resume training from.')
    parser.add_argument('--epochs', type=int, default=2,
                       help='Number of training epochs.')
    parser.add_argument('--dataset_root', type=str,
                       help='Root directory of the dataset images.',
                       default=r'C:\Users\antonio\Desktop\dataset tesi')
    parser.add_argument('--csv_path', type=str,
                       help='Path to the CSV file with labels.',
                       default=r'C:\Users\antonio\Desktop\dataset tesi\merged_labels.csv')
    parser.add_argument('--num_layers_to_unfreeze', type=str, default=0,
                       help='Number of layers to unfreeze beside the attention pooling layer')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training.')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate for optimizer.')

    args = parser.parse_args()

    print(f'args = {args}')

    if not torch.cuda.is_available():
        print("CUDA is not available. Run on a machine with a GPU.")
        sys.exit(1)

    train(
        version=args.version,
        epochs=args.epochs,
        dataset_root_dir=args.dataset_root,
        csv_path = args.csv_path,
        num_layers_to_unfreeze= args.epochs,
        ckpt_path= args.ckpt_path,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        resume_from_ckpt=args.resume_from_ckpt
    )


if __name__ == '__main__':
    main()


