import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Type
import uuid

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from dotenv import load_dotenv
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torchvision.utils import save_image

# Environment and Path Setup
load_dotenv()
REPO_PATH = os.getenv("REPO_PATH")
if REPO_PATH:
    sys.path.append(REPO_PATH)

from probing.probe import Probe
from utils.commons import log_to_disk, get_backbone
from utils.datasets import get_split, resample
from config.task_config import TaskConfig

NUM_WORKERS = 8

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

class Trainer:
    """Encapsulates the training, validation, and testing logic for a probe."""
    def __init__(self, config: TaskConfig, args: argparse.Namespace, device: str = 'cuda'):
        self.config = config
        self.args = args
        self.device = device
        self.probing_type = 'ap' if args.probe_type == 'attention' else 'lp'
        self.version_name = 'Siglip2' if 'google' in args.version else args.version
        self.saving_interval = 10
        self._setup_model_and_optimizer()
        self._setup_datasets()

    def _setup_model_and_optimizer(self):
        """Initializes the model, optimizer, and scheduler."""
        print(f"\n--- Setting up model for {self.args.task} ---")
        backbone, self.transform, hidden_size = get_backbone(self.args.version, self.args.ckpt_path)
        
        self.probe = Probe(
            backbone,
            hidden_size,
            n_out_classes=self.config.n_classes,
            attention_probe=(self.probing_type == 'ap')
        ).to(self.device)

        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.probe.parameters()),
            lr=self.args.learning_rate
        )
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2)
        
        self.start_epoch = 0
        if self.args.resume_from_ckpt:
            if os.path.exists(self.args.resume_from_ckpt):
                self.start_epoch = self.probe.load(self.args.resume_from_ckpt, self.optimizer, self.scheduler, self.device)
            else:
                print(f"Checkpoint file not found at {self.args.resume_from_ckpt}, starting from scratch.")

        

    def _setup_datasets(self):
        """Initializes the datasets and prints distribution info."""
        train_df, val_df = get_split(self.args.csv_path, stratify_column=self.config.stratify_column, test_split=0.1)
        print(f'[DATASET] Training Set Distribution:\n{train_df[self.config.stratify_column].value_counts()}')
        print(f'\n[DATASET] Validation Set Distribution:\n{val_df[self.config.stratify_column].value_counts()}')

        self.dataset_train = self.config.dataset_class(
            root_dir=self.args.dataset_root, df=train_df, transform=self.transform, stratify_on = self.config.stratify_column
        )
        self.dataset_val = self.config.dataset_class(
            root_dir=self.args.dataset_root, df=val_df, transform=self.transform, return_path=True, stratify_on = self.config.stratify_column
        )
    
        if self.config.use_inverse_weights:
            weights = self.dataset_train.get_inverse_weight().to('cuda')
            print(f'Using Inverse weight for CrossEntropyLoss\nWeight = {weights}')
            self.config.criterion = nn.CrossEntropyLoss(weight=weights)

    def _get_save_path(self, epoch: int, is_head_only: bool = False) -> Path:
        """Generates a consistent path for saving checkpoints."""
        task_name = self.args.task.split('_')[0]
        suffix = "head" if is_head_only else str(epoch)
        filename = f"{self.probing_type}_{task_name}_{self.version_name}_{suffix}.pt"
        return self.config.output_folder / filename

    def train(self):
        """Runs the main training loop."""
        print(f"\n--- Starting {self.args.task} Probing ({self.probing_type}) ---")
        self.config.output_folder.mkdir(exist_ok=True)
        minimum_val_loss = float('inf')

        if self.args.task == 'emotion':
            train_loader, val_loader = resample(self.dataset_train, self.dataset_val, self.args.batch_size,
                                                self.config.target_samples_per_class_train, self.config.target_samples_per_class_val,
                                                num_workers=NUM_WORKERS)

        for i in range(self.start_epoch, self.args.epochs):
            print(f"\nEpoch {i+1}/{self.args.epochs}")
            
            # Resample dataloaders for each epoch
            if self.args.task == 'age_classification' or self.args.task == 'gender':
                train_loader, val_loader = resample(self.dataset_train, self.dataset_val, self.args.batch_size,
                                                    self.config.target_samples_per_class_train, self.config.target_samples_per_class_val,
                                                    num_workers=NUM_WORKERS)
            
            train_loss = self._train_epoch(train_loader, i)
            val_loss, cm, class_labels, accuracy = self._evaluate(val_loader)
            self.save_cm(cm, class_labels)

            lr = self.scheduler.get_last_lr()[0]
            log_to_disk(self.config.output_folder, f'{i+1},{train_loss:.5f},{val_loss:.5f},{lr}', f'{self.probing_type}_{self.version_name}')
            print(f'\nFinished Epoch {i+1}\nTraining Loss = {train_loss:.5f}\nValidation Loss = {val_loss:.5f}\nValidation Accuracy = {accuracy:.5f}')

            # Checkpoint saving logic
            if (i + 1) % self.saving_interval == 0 or val_loss < minimum_val_loss or (i + 1) == self.args.epochs:
                if val_loss < minimum_val_loss:
                    print(f"New best validation loss: {val_loss:.5f}. Saving model.")
                    minimum_val_loss = val_loss
                
                save_path = self._get_save_path(epoch=i + 1)
                self.probe.save(path=str(save_path), epoch=i + 1, optimizer=self.optimizer, scheduler=self.scheduler)

        # Save the final probe head
        head_save_path = self._get_save_path(epoch=self.args.epochs, is_head_only=True)
        self.probe.save_head(str(head_save_path))

    def _train_epoch(self, loader: DataLoader, epoch: int) -> float:
        """Performs one training epoch."""
        self.probe.train()
        running_loss = 0.0
        pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Training Epoch {epoch+1}")

        for i, (images, labels) in pbar:
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.probe(images)
            loss = self.config.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            if self.scheduler:
                self.scheduler.step(epoch + i / len(loader))
                lr = self.scheduler.get_last_lr()[0]
                pbar.set_postfix(loss=loss.item(), lr=f"{lr:.6f}")
            else:
                pbar.set_postfix(loss=loss.item())
        
        return running_loss / len(loader)
    
    # age_group = ["0-9","10-19","20-39","40-59","60+"]
    # age_group = ["0-2","3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]
    def _evaluate(self, loader: DataLoader, description: str = "Validating", convert=False, save_misclass=-1):
        """Evaluates the model on a given dataloader."""
        self.probe.eval()
        running_loss = 0.0
        all_true_labels, all_pred_labels = [], []

        main_output_dir = "extreme_misclassifications"
        os.makedirs(main_output_dir, exist_ok=True)

        class_labels = self.config.labels 
        if convert:
            class_labels = ["0-9","10-19","20-39","40-59","60+"]

        label_to_index_map = {label: i for i, label in enumerate(class_labels)}
        pbar = tqdm(loader, desc=description)
        with torch.no_grad():
            for images, labels, path in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.probe(images)
                loss = self.config.criterion(outputs, labels)
                running_loss += loss.item()
                pbar.set_postfix(loss=loss.item())

                # Use the configured functions to process predictions and labels
                pred_labels = self.config.get_predictions(outputs, self.config.labels )
                true_labels = self.config.true_label_map(labels.cpu().numpy().flatten(), self.config.labels)
                if convert:
                    pred_labels = convert_labels(pred_labels)
                    true_labels = convert_labels(true_labels)

                all_pred_labels.extend(pred_labels)
                all_true_labels.extend(true_labels)
                if save_misclass != -1:
                    for i, (true_str, pred_str, path) in enumerate(zip(true_labels, pred_labels, path)):
                        # NEW: Convert the string labels to their integer indices using the map
                        true_idx = label_to_index_map[true_str]
                        pred_idx = label_to_index_map[pred_str]
                        
                        # Now, this check works correctly with the indices
                        if abs(true_idx - pred_idx) >= save_misclass:
                            
                            true_folder_str = true_str.replace('+', 'plus')
                            pred_folder_str = pred_str.replace('+', 'plus')
                            
                            error_folder_name = f"true_{true_folder_str}_pred_{pred_folder_str}"
                            specific_output_dir = os.path.join(main_output_dir, error_folder_name)
                            os.makedirs(specific_output_dir, exist_ok=True)
                            
                            image_to_save = images[i]


                            image_filename = path.split('/')[-1]
                            full_save_path = os.path.join(specific_output_dir, image_filename)
                            save_image(image_to_save, full_save_path)

        accuracy = accuracy_score(all_true_labels, all_pred_labels) * 100
        cm = confusion_matrix(all_true_labels, all_pred_labels, labels=class_labels)
        return running_loss / len(loader), cm, class_labels, accuracy

    def test(self):
        """Performs final evaluation on the test set."""
        print(f"\n--- Final Testing on {self.args.task} ---\nSamples from {self.config.test_set_path}")
        test_df = pd.read_csv(self.config.test_set_path)
        test_set = self.config.dataset_class(
            root_dir=self.args.dataset_root, df=test_df, transform=self.transform, return_path=True
        )
        test_loader = DataLoader(test_set, batch_size=self.args.batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=False)

        test_loss, cm, class_labels, accuracy = self._evaluate(test_loader, description="Final Testing", convert=False, save_misclass=0)
        
        print(f"Final Test Loss: {test_loss:.5f}\nFinal accuracy: {accuracy:.3f}%")
        print("Confusion Matrix:\n", cm)

        # Plot and save confusion matrix
        self.save_cm(cm, class_labels)
        
        log_to_disk(self.config.output_folder, f'test_loss,{test_loss:.5f},{accuracy:.5f}', f'{self.probing_type}_{self.version_name}')

    def save_cm(self, cm, class_labels):
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
        disp.plot()
        cm_path = self.config.output_folder / f'cm_{self.probing_type}_{self.version_name}.jpg'
        plt.savefig(cm_path, bbox_inches='tight')
        print(f"Confusion matrix saved to {cm_path}")


    def cleanup(self):
        """
        Perform cleanup operations.
        This method is called to ensure resources are released.
        """
        print("Shutting down Trainer and associated workers.")
        self.train_loader = None
        self.test_loader = None