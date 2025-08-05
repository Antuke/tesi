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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
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
from config.task_config import TaskConfig, MTLConfig
from multitask.multitask_probe import MultiTaskProbe

AGE_TASK_NUM = 0
GENDER_TASK_NUM = 1
EMOTION_TASK_NUM = 2
STATIC_TASK_WEIGHT = [1.0,1.0,1.0]
NUM_WORKERS = 8
USE_WEIGHTED_LOSS = ['Age','Emotion']
device = 'cuda'

class Trainer:
    """Encapsulates the training, validation, and testing logic for a MT probe."""
    def __init__(self, config: MTLConfig, args: argparse.Namespace, device: str = 'cuda'):
        self.config = config
        self.args = args
        self.device = device
        self.version_name = 'Siglip2' if 'google' in args.version else args.version
        self.saving_interval = 10
        self.dataset_train = None # it will get initialized in setup methods
        self.transform = None # it will get initialized in setup methods
        self.train_loader = None
        self.val_loader = None
        self._setup_model_and_optimizer()
        self._setup_loaders()
        self.criterions = self._setup_criterions()
        self.probing_type = 'mtl'

    def _setup_criterions(self):
        """Initialize the loss function with weights"""
        weights = self.dataset_train.get_inverse_weights_loss_mc(1)
        criterions = []
        for criterion, task_name in zip(self.config.criterions, self.config.task_names):
            if task_name in USE_WEIGHTED_LOSS:
                print(weights[task_name])
                criterions.append(criterion(weight=weights[task_name].to(device),ignore_index=-100))
            else:
                criterions.append(criterion(ignore_index=-100))

        return criterions

    def _setup_model_and_optimizer(self):
        """Initializes the model, optimizer, and scheduler."""
        print(f"\n--- Setting up model for MTL  ---")
        backbone, transform, hidden_size = get_backbone(self.args.version, self.args.ckpt_path)
        self.transform = transform
        self.mtl_probe = MultiTaskProbe(backbone=backbone,
                               backbone_output_dim=hidden_size,
                               num_layers_to_unfreeze=0).to(device)

        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.mtl_probe.parameters()),
            lr=self.args.learning_rate
        )
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2)
        
        self.start_epoch = 0
        if self.args.resume_from_ckpt:
            if os.path.exists(self.args.resume_from_ckpt):
                self.start_epoch = self.mtl_probe.load(self.args.resume_from_ckpt,optimizer = self.optimizer,scheduler = self.scheduler)
            else:
                print(f"Checkpoint file not found at {self.args.resume_from_ckpt}, starting from scratch.")

    def _setup_loaders(self):
        """Initializes the dataloaders with appropriate sampling to ensure balance in training. To call at the start of each epoch"""
        if self.dataset_train == None:
            self.dataset_train = self.config.dataset_class(
                root_dir=self.args.dataset_root, 
                gender_csv_path=self.args.csv_path_gender, 
                emotion_csv_path=self.args.csv_path_emotions, 
                transform=self.transform
            )
        else:
            self.dataset_train.new_epoch_resample()
        sampler_weights = self.dataset_train.get_sampler_weights()

        # the dataset is heavily imbalanced towards samples with age and genders
        # so we need a weighted random sampler
        sampler = WeightedRandomSampler(
            weights=sampler_weights,
            num_samples=len(sampler_weights),
            replacement=True
        )

        self.train_loader = DataLoader(self.dataset_train, sampler=sampler,
                                batch_size=self.args.batch_size,num_workers=NUM_WORKERS, pin_memory=True)


        dataset_val = self.config.dataset_class(
            root_dir=self.args.dataset_root, 
            gender_csv_path=self.config.test_set_gender_age, 
            emotion_csv_path=self.config.test_set_emotions, 
            transform=self.transform,
            keep=1.0
        )

        # no need of a special sampler for validation 
        self.val_loader = DataLoader(dataset_val, batch_size=self.args.batch_size,num_workers=NUM_WORKERS, pin_memory=True)


    def _get_save_path(self, epoch: int, is_head_only: bool = False) -> Path:
        """Generates a consistent path for saving checkpoints."""
        task_name = 'mtl'
        suffix = f"head+{str(epoch)}" if is_head_only else str(epoch)
        filename = f"{task_name}_{self.version_name}_{suffix}.pt"
        return self.config.output_folder / 'ckpt' / filename
    

    def train(self):
        """Runs the main training loop."""
        print(f"\n--- Starting Training ---")
        self.config.output_folder.mkdir(exist_ok=True)
        (self.config.output_folder / 'ckpt').mkdir(exist_ok=True)
        minimum_val_loss = float('inf')


        for i in range(self.start_epoch, self.args.epochs):
            print(f"\nEpoch {i+1}/{self.args.epochs}")
            
            self._setup_loaders()
            
            train_loss_avg, train_loss_for_tasks = self._train_epoch(self.train_loader, i)
            val_loss_avg, val_loss_for_tasks, cms, accuracy_avg, accuracies = self._evaluate(self.val_loader)
            for j, cm in enumerate(cms):
                self.save_cm(cm, class_labels=self.config.class_labels_list[j] , task = self.config.task_names[j])

            lr = self.scheduler.get_last_lr()[0]
            accuracies_string = ",".join(f"{x:.3f}" for x in accuracies)
            loss_string_train_tasks = ",".join(f"{x:.4f}" for x in train_loss_for_tasks)
            loss_string_val_tasks = ",".join(f"{x:.4f}" for x in val_loss_for_tasks)
            log_string=f'{i+1},{train_loss_avg:.4f},{loss_string_train_tasks},{val_loss_avg:.4f},{loss_string_val_tasks},{accuracy_avg:.3f},{accuracies_string},{lr:.6f}'
            log_to_disk(self.config.output_folder, log_string,
                         f'{self.probing_type}_{self.version_name}',header=self.config.header)
            print(f'\nFinished Epoch {i+1}\nTraining Loss = {train_loss_avg:.4f}\nValidation Loss = {val_loss_avg:.4f}\nValidation Accuracy = {accuracy_avg:.4f}')

            # Checkpoint saving logic
            if (i + 1) % self.saving_interval == 0 or val_loss_avg < minimum_val_loss or (i + 1) == self.args.epochs:
                if val_loss_avg < minimum_val_loss:
                    print(f"New best validation loss: {val_loss_avg:.5f}. Saving model.")
                    minimum_val_loss = val_loss_avg
                
                save_path = self._get_save_path(epoch=i + 1)
                self.mtl_probe.save(path=str(save_path), epoch=i + 1, optimizer=self.optimizer, scheduler=self.scheduler)

 

    def _train_epoch(self, loader: DataLoader, epoch: int) -> float:
        """Performs one training epoch. Returns avg loss of epoch and a list 
        containing the avg loss for each task """
        self.mtl_probe.train()
        running_loss = 0.0

        num_tasks = len(self.criterions)
        task_losses = [0.0] * num_tasks
        samples_per_task = [0] * num_tasks

        pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Training Epoch {epoch+1}")

        for i, (images, labels) in pbar:
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()

            # age_logits, gender_logits, emotion_logits
            task_outputs = self.mtl_probe(images)

            loss = 0.0
            current_batch_task_losses = [0.0] * num_tasks

            for task_idx, criterion in enumerate(self.criterions):
                current_output = task_outputs[task_idx]
                current_target = labels[:, task_idx]
                if (current_target == -100).all():
                    continue
                
                # how many sample for current task are labelled
                samples_per_task[task_idx] += (current_target != -100).sum().item()
                
                task_loss = criterion(current_output, current_target)
                
                current_batch_task_losses[task_idx] = task_loss.item()
                
                task_losses[task_idx] += current_batch_task_losses[task_idx]

                loss += task_loss * STATIC_TASK_WEIGHT[task_idx]

            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            avg_running_loss = running_loss / (i + 1)

            task_loss_str = ", ".join([f"{self.config.task_names[i]}: {l:.4f}" for i, l in enumerate(current_batch_task_losses)])
            pbar.set_description(
                f"Training - Avg Loss: {avg_running_loss:.4f} | Batch Task Losses: [{task_loss_str}]"
            )

            # update lr 
            if self.scheduler:
                self.scheduler.step(epoch + i / len(loader))
                lr = self.scheduler.get_last_lr()[0]
                pbar.set_postfix(loss=loss.item(), lr=f"{lr:.6f}")
            else:
                pbar.set_postfix(loss=loss.item())

        avg_epoch_loss = running_loss / len(loader)
        avg_task_losses = [
            total_loss / num_samples
            for total_loss, num_samples in zip(task_losses, samples_per_task)
        ]
        
        return avg_epoch_loss, avg_task_losses

    def _evaluate(self, loader: DataLoader, description: str = "Validating", convert=False, save_misclass=-1):
        """Evaluates the model on a given dataloader."""
        self.mtl_probe.eval()
        running_loss = 0.0
        task_losses = [0.0] * len(self.criterions)
        num_tasks = len(self.criterions)
        all_gender_preds, all_gender_true = [], []
        all_age_preds, all_age_true = [], []
        all_emotion_preds, all_emotion_true = [], []

        main_output_dir = "extreme_misclassifications"
        os.makedirs(main_output_dir, exist_ok=True)
        num_gender_classes = range(len(self.config.class_labels_list[GENDER_TASK_NUM]))
        num_age_classes = range(len(self.config.class_labels_list[AGE_TASK_NUM]))
        num_emotion_classes = range(len(self.config.class_labels_list[EMOTION_TASK_NUM]))


        pbar = tqdm(loader, desc=description)
        with torch.no_grad():

            for i , (images, labels) in enumerate(pbar):
                images, labels = images.to(self.device), labels.to(self.device)
                
                task_outputs = self.mtl_probe(images)
                current_batch_task_losses = [0.0] * num_tasks
                batch_total_loss = 0.0

                for task_idx, criterion in enumerate(self.criterions):
                    current_output = task_outputs[task_idx]
                    current_target = labels[:, task_idx]
                    task_loss = criterion(current_output, current_target)
                    

                    valid_mask = (current_target != -100)
                    if not valid_mask.any():
                        continue
                    
                    task_losses[task_idx] += task_loss.item()
                    batch_total_loss += task_loss * STATIC_TASK_WEIGHT[task_idx]
                    current_batch_task_losses[task_idx] = task_loss.item()
                    
                    if task_idx == GENDER_TASK_NUM:
                        preds = current_output[valid_mask].argmax(dim=1)
                        all_gender_preds.extend(preds.cpu().numpy())
                        all_gender_true.extend(current_target[valid_mask].cpu().numpy())
                   
                    elif task_idx == AGE_TASK_NUM:
                        preds = current_output[valid_mask].argmax(dim=1)
                        all_age_preds.extend(preds.cpu().numpy())
                        all_age_true.extend(current_target[valid_mask].cpu().numpy())

                    elif task_idx == EMOTION_TASK_NUM:
                        preds = current_output[valid_mask].argmax(dim=1)
                        all_emotion_preds.extend(preds.cpu().numpy())
                        all_emotion_true.extend(current_target[valid_mask].cpu().numpy())

                running_loss += batch_total_loss.item()
                
                task_loss_str = ", ".join([f"{self.config.task_names[j]}: {l:.4f}" for j, l in enumerate(current_batch_task_losses)])
                avg_running_loss = running_loss / (i + 1)
                pbar.set_description(
                    f"Validating - Avg Loss: {avg_running_loss:.4f} | Batch Task Losses: [{task_loss_str}]"
                )

        # Calculate final metrics
        avg_loss = running_loss / len(loader)
        avg_task_losses = [l / len(loader) for l in task_losses]

        # Create confusion matrices
        cm_gender = confusion_matrix(all_gender_true, all_gender_preds, labels=num_gender_classes)
        cm_age = confusion_matrix(all_age_true, all_age_preds, labels=num_age_classes)
        cm_emotion = confusion_matrix(all_emotion_true, all_emotion_preds, labels=num_emotion_classes)

        # accuracies scores
        accuracy_gender = accuracy_score(all_gender_true, all_gender_preds) * 100
        accuracy_age = accuracy_score(all_age_true,all_age_preds) * 100
        accuracy_emotion = accuracy_score(all_emotion_true, all_emotion_preds) * 100

        return avg_loss, avg_task_losses, (cm_age, cm_gender, cm_emotion), (accuracy_gender + accuracy_age + accuracy_emotion)/3, (accuracy_age, accuracy_gender, accuracy_emotion)
    
    def save_cm(self, cm, class_labels, final=False, task=None):
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
        disp.plot()
        cm_path = self.config.output_folder / f'cm_{self.probing_type}_{self.version_name}_{task}.jpg'
        if final:
            cm_path = self.config.output_folder / f'cm_{self.probing_type}_{self.version_name}_{task}_test_set.jpg'
        plt.savefig(cm_path, bbox_inches='tight')
        print(f"Confusion matrix saved to {cm_path}")

    def cleanup(self):
        print("\n[Trainer Cleanup] Searching for active worker processes...")
        import multiprocessing
        active_procs = multiprocessing.active_children()
        
        if not active_procs:
            print("[Trainer Cleanup] No active child processes found.")
            return

        for process in active_procs:
            print(f"[Trainer Cleanup] Shutting down process: {process.name} (PID: {process.pid})")
            process.terminate()  # Sends a SIGTERM signal to the process
            process.join(timeout=5) # Waits for the process to terminate gracefully
            if process.is_alive():
                print(f"[Trainer Cleanup] Process {process.pid} did not terminate, forcing kill.")
                process.kill() # If it doesn't terminate, force kill it with SIGKILL
        
        print("[Trainer Cleanup] All worker processes have been handled.")