import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Type, Tuple
import uuid
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from dotenv import load_dotenv
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import torch.optim as optim
from torch.optim.lr_scheduler import  CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image

# Environment and Path Setup
load_dotenv()
REPO_PATH = os.getenv("REPO_PATH")
if REPO_PATH:
    sys.path.append(REPO_PATH)

from probing.probe import Probe
from utils.commons import log_to_disk, get_backbone, convert_labels
from utils.datasets import get_split, resample
from config.task_config import SingleConfig

NUM_WORKERS = 8





class Trainer:
    """Encapsulates the training, validation, and testing logic for a probe."""
    def __init__(self, config: SingleConfig, args: argparse.Namespace):
        self.config = config
        self.args = args
        self.task = config.task
        self.device = torch.device(config.device)
        self.probing_type = 'ap' if args.probe_type == 'attention' else 'lp'
        self.version_name = 'Siglip2-base-patch16-224' if 'google' in args.version else args.version
        self.saving_interval = 10
        self.start_epoch = 0
        self.config.output_folder = self.config.output_folder / self.version_name
        self.ckpt_folder = 'ckpt_' + self.probing_type
        # Components to be initialized in self.setup()
        self.probe = None
        self.transform = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.best_path = None
        self.setup()
        
    def setup(self):
        self._setup_model_and_optimizer()
        self._setup_loaders()
        self._setup_criterion()

    def _setup_model_and_optimizer(self):
        """Initializes the model, optimizer, and scheduler."""
        print("[TRAINER] Setting up model and optimizer...")
        backbone, self.transform, hidden_size = get_backbone(self.args.version, self.args.ckpt_path)
        
        self.probe = Probe(
            backbone,
            hidden_size,
            n_out_classes=self.task.num_classes,
            attention_probe=(self.probing_type == 'ap')
        ).to(self.device)

        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.probe.parameters()),
            lr=self.args.learning_rate
        )
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2)
        
        if self.args.resume_from_ckpt and os.path.exists(self.args.resume_from_ckpt):
            print(f"[TRAINER] Resuming from checkpoint: {self.args.resume_from_ckpt}")
            self.start_epoch = self.probe.load(self.args.resume_from_ckpt, self.optimizer, self.scheduler, self.device)
        else:
            print("[TRAINER] Starting from scratch.")

        

    def _setup_loaders(self):
        """Initializes the datasets and dataloaders."""
        print("[TRAINER] Setting up data loaders...")
        train_df, val_df = get_split(self.config.csv_path, stratify_column=self.task.stratify_column, test_split=0.1)

        dataset_train = self.task.dataset_class(
            root_dir=self.config.dataset_root, df=train_df, transform=self.transform, stratify_on=self.task.stratify_column
        )
        dataset_val = self.task.dataset_class(
            root_dir=self.config.dataset_root, df=val_df, transform=self.transform, return_path=True, stratify_on=self.task.stratify_column
        )


        if self.task.target_samples_per_class_train > 0:
            print("[TRAINER] Using resampling for train and validation loaders.")
            self.train_loader, self.val_loader = resample(
                dataset_train, dataset_val, self.args.batch_size,
                self.task.target_samples_per_class_train,
                self.task.target_samples_per_class_val,
                num_workers=self.config.num_workers
            )
        else:
            print("[TRAINER] Using standard data loaders.")
            self.train_loader = DataLoader(dataset_train, batch_size=self.args.batch_size, shuffle=True, num_workers=self.config.num_workers, pin_memory=True)
            self.val_loader = DataLoader(dataset_val, batch_size=self.args.batch_size, num_workers=self.config.num_workers, pin_memory=True)

        # Store dataset for potential use in criterion setup
        self.dataset_train = dataset_train
        self.dataset_val = dataset_val
    
    def _sample_loaders(self):
        self.train_loader, self.val_loader = resample(
                self.dataset_train, self.dataset_val, self.args.batch_size,
                self.task.target_samples_per_class_train,
                self.task.target_samples_per_class_val,
                num_workers=self.config.num_workers
            )

    def _setup_test_loader(self):
        """Initializes the test dataloader when needed."""
        print("[TRAINER] Setting up test data loader...")
        test_df = pd.read_csv(self.config.test_csv_path)
        dataset_test = self.task.dataset_class(
            root_dir=self.config.dataset_root, df=test_df, transform=self.transform, return_path=True
        )
        self.test_loader = DataLoader(dataset_test, batch_size=self.args.batch_size, num_workers=self.config.num_workers, pin_memory=True)

    def _setup_criterion(self):
        """Initializes the loss function with optional class weighting."""
        print("[TRAINER] Setting up criterion...")
        params = {}
        if self.task.use_weighted_loss:
            weights = self.dataset_train.get_inverse_weight().to(self.device)
            params['weight'] = weights
            print(f"[TRAINER] Using weighted loss for task: {self.task.name}")
            print(f"Using the following weights = {weights.tolist()}")
        self.criterion = self.task.criterion(**params)

    def train(self):
        """Runs the main training loop."""
        
        print(f"\n--- Starting {self.task.name} Probing ({self.probing_type}) ---")
        self.config.output_folder.mkdir(parents=True, exist_ok=True)
        (self.config.output_folder / 'confusion_matricies').mkdir(exist_ok=True)
        (self.config.output_folder / self.ckpt_folder ).mkdir(exist_ok=True)
        
        maximum_accuracy = -float('inf')

        for epoch in range(self.start_epoch, self.args.epochs):
            print(f"\nEpoch {epoch + 1}/{self.args.epochs}")
            if self.task.target_samples_per_class_train > 0:
                self._sample_loaders()
            train_results = self._run_epoch(self.train_loader, is_training=True, epoch_idx=epoch)
            val_results = self._run_epoch(self.val_loader, is_training=False, description="Validating")

            print(f"TRAINING Loss: {train_results['loss']:.5f}")
            print(f"VALIDATION Loss: {val_results['loss']:.5f}, Accuracy: {val_results['accuracy']:.3f}%")
            
            self.save_confusion_matrix(val_results['confusion_matrix'], epoch)
            self._log_epoch_results(epoch, train_results, val_results)
            
            val_accuracy = val_results['accuracy']
            if (epoch + 1) % self.saving_interval == 0 or val_accuracy > maximum_accuracy or (epoch + 1) == self.args.epochs:
                if val_accuracy > maximum_accuracy:
                    print(f"[TRAINER] New best validation loss: {val_accuracy:.5f}. Saving model.")
                    maximum_accuracy = val_accuracy
                    if self.best_path:  
                        if os.path.exists(self.best_path):
                            os.remove(self.best_path)
                    self.best_path = self.config.output_folder / self.ckpt_folder / f"{self.probing_type}_{self.task.name}_{self.version_name}_{epoch + 1}.pt"
                
                save_path = self.config.output_folder / self.ckpt_folder / f"{self.probing_type}_{self.task.name}_{self.version_name}_{epoch + 1}.pt"
                self.probe.save(path=str(save_path), epoch=epoch + 1, optimizer=self.optimizer, scheduler=self.scheduler)

    def test(self, ckpt_path=None, k=2):
        """Evaluates the model on the test set. Supports the computing of top-k accuracy"""

        self._setup_test_loader()
        best_epoch = self.probe.load(self.best_path)
        if ckpt_path:
            self.probe.load(ckpt_path)

        print(f"\n--- Starting Final Testing for {self.task.name} ---")
        
        save_extreme_wrong = False
        if self.task.name == 'Age':
            save_extreme_wrong = True
        
        if self.task.name == 'Gender':
            k = 1

        test_results = self._run_epoch(
            self.test_loader,
            is_training=False,
            description="Testing",
            k=k,
            save_extreme_wrong=save_extreme_wrong
        )

        self.save_confusion_matrix(test_results['confusion_matrix'], is_final=True)

        print("\n--- Test Results ---")
        print(f"Average Loss: {test_results['loss']:.4f}")
        print(f"Final Accuracy: {test_results['accuracy']:.2f}%")
        print(f"Top-{k} Accuracy: {test_results['top_k_accuracy']:.2f}%")
        print("Confusion Matrix:\n", test_results['confusion_matrix'])
        self._log_epoch_results(best_epoch, None, test_results, k=k)


    def _run_epoch(self, loader: DataLoader, is_training: bool, description: str = "Training", epoch_idx: int = 0, k: int = 1, save_extreme_wrong: bool = False) -> Dict[str, Any]:
        """Runs a single epoch of training or evaluation."""
        self.probe.train(is_training)

        total_loss = 0.0
        all_valid_preds, all_valid_true = [], []
        correct_top_k = 0
        total_valid_samples = 0
        pbar = tqdm(loader, desc=f"{description} Epoch {epoch_idx+1 if is_training else ''}")

        context = torch.no_grad() if not is_training else torch.enable_grad()
        with context:
            for i, batch in enumerate(pbar):
                if save_extreme_wrong or not is_training: # Path is needed for testing/validation
                     images, labels, paths = batch
                else:
                     images, labels = batch
                     paths = [None] * len(labels)

                loss, outputs = self._process_batch(images, labels)

                if is_training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    if self.scheduler:
                        self.scheduler.step(epoch_idx + i / len(loader))

                total_loss += loss.item()
                pbar.set_postfix(loss=loss.item())

                if not is_training:
                    valid_labels_mask = labels != -100
                    
                    # If there are no valid labels in this batch, skip to the next one
                    if not torch.any(valid_labels_mask):
                        continue

                    valid_labels = labels[valid_labels_mask].to(self.device)
                    valid_outputs = outputs[valid_labels_mask]
                    valid_paths = [path for idx, path in enumerate(paths) if valid_labels_mask[idx]]
                    
                    pred_indices = valid_outputs.argmax(dim=1)
                    
                    all_valid_preds.extend(pred_indices.cpu().numpy())
                    all_valid_true.extend(valid_labels.cpu().numpy())

                    _, top_k_preds = torch.topk(valid_outputs, k, dim=1)
                    expanded_labels = valid_labels.view(-1, 1).expand_as(top_k_preds)
                    correct_top_k += torch.sum(torch.any(top_k_preds == expanded_labels, dim=1)).item()
                    total_valid_samples += valid_labels.size(0)

                    if save_extreme_wrong:
                        misclassified = pred_indices != valid_labels
                        for idx in torch.where(misclassified)[0]:
                            true_label = valid_labels[idx].item()
                            pred_label = pred_indices[idx].item()

                            if abs(true_label - pred_label) > 3:
                                output_dir = self.config.output_folder / f"extreme_wrong_classifications/true_{true_label}/pred_{pred_label}"
                                output_dir.mkdir(parents=True, exist_ok=True)
                                
                                original_image_tensor = images[valid_labels_mask][idx]
                                img_path = Path(valid_paths[idx])
                                save_image(original_image_tensor, output_dir / f"{img_path.stem}_{uuid.uuid4()}.png")

        results = {'loss': total_loss / len(loader)}
        if not is_training:
            results['accuracy'] = accuracy_score(all_valid_true, all_valid_preds) * 100 if all_valid_true else 0
            results['confusion_matrix'] = confusion_matrix(all_valid_true, all_valid_preds, labels=range(self.task.num_classes))
            results['top_k_accuracy'] = (correct_top_k / total_valid_samples) * 100 if total_valid_samples > 0 else 0
            
        return results

    def _process_batch(self, images: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Processes a single batch of data, returning loss and model outputs."""
        images, labels = images.to(self.device), labels.to(self.device)
        outputs = self.probe(images)
        loss = self.criterion(outputs, labels)
        return loss, outputs

    def _log_epoch_results(self, epoch: int, train_results: Dict, val_results: Dict, k: int = 1):
        """Logs the results of an epoch to a CSV file."""
        lr = self.scheduler.get_last_lr()[0] if self.scheduler else self.args.learning_rate
        if train_results:
            header = self.config.header
            log_string = (
                f"{epoch + 1},"
                f"{train_results['loss']:.5f},"
                f"{val_results['loss']:.5f},"
                f"{val_results['accuracy']:.3f},"
                f"{lr:.6f}"
            )
        else: 
            header = f"epoch,test_loss,test_accuracy,top_{k}_accuracy,learning_rate"
            log_string = (
                f"{epoch + 1},"
                f"{val_results['loss']:.5f},"
                f"{val_results['accuracy']:.3f},"
                f"{val_results.get('top_k_accuracy', 0):.3f},"
                f"{lr:.6f}"
            )

        log_to_disk(
            self.config.output_folder,
            log_string,
            f'{self.probing_type}_{self.version_name}_testing' if not train_results else f'{self.probing_type}_{self.version_name}',
            header=header
        )

    def save_confusion_matrix(self, cm: np.ndarray, epoch: int = 0, is_final: bool = False):
        """Saves a confusion matrix plot to disk."""
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.task.class_labels)
        disp.plot(xticks_rotation='vertical')
        
        if is_final:
            filename = f'cm_{self.probing_type}_{self.version_name}_test_set.jpg'
        else:
            filename = f'cm_{self.probing_type}_{self.version_name}_epoch_{epoch+1}.jpg'
        
        cm_path = self.config.output_folder / 'confusion_matricies' / filename
        plt.savefig(cm_path, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix saved to {cm_path}")

    @staticmethod
    def cleanup():
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