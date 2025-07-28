# Code to independently train and validate the tree attention probe
import argparse
import sys
import os

from sklearn.metrics import confusion_matrix
from tqdm import tqdm
REPO_PATH = "C:/Users/antonio/Desktop/perception_models/"
sys.path.append(REPO_PATH)


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
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR
from probe import Probe

####---------- CONFIGS ----------####

QUICK = True
AGE_FOLDER = './age_outputs'
GENDER_FOLDER = './gender_outputs'
EMOTION_FOLDER = './emotions_outputs'
age_labels_name = ["0-2","0-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "more than 70"]
gender_labels_name = ["Male", "Female"]
emotion_labels_name = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


TASK_CONFIG = {
    'age_regression': {
        'n_classes': 1,
        'criterion': nn.MSELoss(),
        'dataset_class': AgeDataset,
        'dataset_kwargs': {'classification': False},
        'labels': age_labels_name,
        'output_folder': AGE_FOLDER
    },
    'age_classification': {
        'n_classes': len(age_labels_name),
        'criterion': nn.CrossEntropyLoss(),
        'dataset_class': AgeDataset,
        'dataset_kwargs': {'classification': True},
        'labels': age_labels_name,
        'output_folder': AGE_FOLDER
    },
    'gender': {
        'n_classes': 1,
        'criterion': nn.BCEWithLogitsLoss(),
        'dataset_class': GenderDataset,
        'dataset_kwargs': {},
        'labels': gender_labels_name,
        'output_folder': GENDER_FOLDER
    },
    'emotion': {
        'n_classes': len(emotion_labels_name),
        'criterion': nn.CrossEntropyLoss(),
        'dataset_class': EmotionDataset,
        'dataset_kwargs': {},
        'labels': emotion_labels_name,
        'output_folder': EMOTION_FOLDER
    }
}



def get_categorical_age(age: float) -> str:
    """Converts a continuous age into a categorical age group."""
    if age < 10: return "0-9"
    if age < 20: return "10-19"
    if age < 30: return "20-29"
    if age < 40: return "30-39"
    if age < 50: return "40-49"
    if age < 60: return "50-59"
    if age < 70: return "60-69"
    return "70+"






def train_epoch(loader : DataLoader, probe: Probe, criterion, optimizer: Optimizer, scheduler: _LRScheduler):
    probe.train()
    running_loss = 0.0
    loader = tqdm(loader, desc="Training")

    for images, labels in loader:
        images = images.to('cuda')
        labels = labels.to('cuda')

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = probe(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        loader.set_postfix(loss=loss.item())
        
    scheduler.step()
    return running_loss / len(loader)

def validate(loader: DataLoader, probe: Probe, criterion, task: str):
    probe.eval()
    running_loss = 0.0
    loader_tqdm = tqdm(loader, desc="Validating")

    all_true_labels = []
    all_pred_labels = []


    with torch.no_grad():
        for images, labels in loader_tqdm:
            images, labels = images.to('cuda'), labels.to('cuda')

            outputs = probe(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            loader_tqdm.set_postfix(loss=loss.item())

            # Process predictions based on the task
            if task == 'age_regression':
                preds_flat = outputs.cpu().numpy().flatten()
                labels_flat = labels.cpu().numpy().flatten()
                all_pred_labels.extend([get_categorical_age(p) for p in preds_flat])
                all_true_labels.extend([get_categorical_age(l) for l in labels_flat])
                class_labels = age_labels_name
            elif task == 'age_classification':
                preds_flat = outputs.argmax(dim=1).cpu().numpy()
                labels_flat = labels.cpu().numpy().flatten()
                all_pred_labels.extend([age_labels_name[p] for p in preds_flat])
                all_true_labels.extend([age_labels_name[l] for l in labels_flat])
                class_labels = age_labels_name
            elif task == 'gender':
                preds_flat = torch.sigmoid(outputs).cpu().numpy().flatten()
                labels_flat = labels.cpu().numpy().flatten()
                all_pred_labels.extend([gender_labels_name[int(p > 0.5)] for p in preds_flat])
                all_true_labels.extend([gender_labels_name[int(l)] for l in labels_flat])
                class_labels = gender_labels_name
            elif task == 'emotion':
                preds_flat = outputs.argmax(dim=1).cpu().numpy()
                labels_flat = labels.cpu().numpy().flatten()
                all_pred_labels.extend([emotion_labels_name[p] for p in preds_flat])
                all_true_labels.extend([emotion_labels_name[l] for l in labels_flat])
                class_labels = emotion_labels_name


    cm = confusion_matrix(all_true_labels, all_pred_labels, labels=class_labels)
    return running_loss / len(loader), cm




def probe_task(
    task: str,
    version: str,
    epochs: int,
    dataset_root_dir: str,
    csv_path: str,
    attention_probing: bool = True,
    ckpt_path: str = None,
    resume_from_ckpt: str = None,
    batch_size: int = 32,
    learning_rate: float = 0.001
    ):
    """
    Probing function for all tasks.

    Args:
        task: One of 'age_regression', 'age_classification', 'gender', 'emotion'
        version: Backbone model version
        epochs: Number of training epochs
        dataset_root_dir: Root directory of dataset images
        csv_path: Path to CSV file with labels
        attention_probing: Whether to use attention probing or linear probing
        ckpt_path: Path to backbone checkpoint
        resume_from_ckpt: Path to a checkpoint to resume training from.
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
    """



    config = TASK_CONFIG[task]
    probing_type = 'ap' if attention_probing else 'lp'

    print(f"--- Starting {task} Probing ---")

    # Setup model
    backbone, transform, hidden_size = get_backbone(version, ckpt_path)
    probe = Probe(
        backbone,
        hidden_size,
        n_out_classes=config['n_classes'],
        attention_probe=attention_probing
    ).to('cuda')

    # Setup dataset
    dataset = config['dataset_class'](
        root_dir=dataset_root_dir,
        csv_file=csv_path,
        transform=transform,
        **config['dataset_kwargs']
    )

    if QUICK:
        dataset = Subset(dataset, range(500))

    train_loader, val_loader, test_loader = get_loaders(
        dataset,
        torch.Generator().manual_seed(42),
        batch_size=batch_size
    )

    # Setup training
    criterion = config['criterion']
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, probe.parameters()),
        lr=learning_rate
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    start_epoch = 0
    if resume_from_ckpt:
        if os.path.exists(resume_from_ckpt):
            start_epoch = probe.load(resume_from_ckpt, optimizer, 'cuda')
        else:
            print(f"Checkpoint file not found at {resume_from_ckpt}, starting from scratch.")


    # Training loop
    for i in range(start_epoch, epochs):
        print(f"\nEpoch {i+1}/{epochs}")
        train_loss = train_epoch(
            loader=train_loader,
            probe=probe,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler
        )
        val_loss, _ = validate(
            loader=val_loader,
            probe=probe,
            criterion=criterion,
            task=task
        )

        # Log results
        log_to_disk(config['output_folder'], f'{i+1},{train_loss:.5f},{val_loss:.5f}', probing_type)

        # Save checkpoint
        task_name = task.split('_')[0]  # 'age', 'gender', 'emotion'
        save_path = f"{config['output_folder']}/{probing_type}_{task_name}"
        probe.save(path=save_path, epoch=i + 1, optimizer=optimizer)


    head_save_path = f"{config['output_folder']}/{probing_type}_{task_name}_final_head_{version}.pt"
    probe.save_head(head_save_path)

    # Final test
    print(f"\n--- Final Testing on {task} ---")
    test_loss, cm = validate(
        loader=test_loader,
        probe=probe,
        criterion=criterion,
        task=task
    )
    print(f"Final Test Loss: {test_loss:.5f}")
    print("Confusion Matrix:\n", cm)

    return test_loss, cm

def main():
    parser = argparse.ArgumentParser(description="Train and validate attention probes for different tasks.")
    parser.add_argument('--task', type=str,
                       choices=['age_regression', 'age_classification', 'gender', 'emotion'],
                       help='Task to perform.', default='gender')
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
    parser.add_argument('--probe_type', type=str, default='attention',
                       choices=['attention', 'linear'],
                       help='Type of probing: "attention" or "linear".')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training.')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate for optimizer.')

    args = parser.parse_args()

    print(f'args = {args}')

    if not torch.cuda.is_available():
        print("CUDA is not available. Run on a machine with a GPU.")
        sys.exit(1)

    attention_probing = args.probe_type == 'attention'

    probe_task(
        task=args.task,
        version=args.version,
        epochs=args.epochs,
        dataset_root_dir=args.dataset_root,
        csv_path=args.csv_path,
        attention_probing=attention_probing,
        ckpt_path=args.ckpt_path,
        resume_from_ckpt=args.resume_from_ckpt,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )


if __name__ == '__main__':
    main()