import argparse
import os
import sys
from pathlib import Path
from typing import Any,  Dict, List, Tuple, Type

import matplotlib.pyplot as plt
import torch
from dotenv import load_dotenv
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import torch.optim as optim
from torch.optim.lr_scheduler import  CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torch.utils.data import DataLoader,  WeightedRandomSampler
from tqdm import tqdm
import numpy as np
from torchvision.utils import save_image
from torch.cuda.amp import autocast
# Environment and Path Setup
load_dotenv()
REPO_PATH = os.getenv("REPO_PATH")
if REPO_PATH:
    sys.path.append(REPO_PATH)

from probing.probe import Probe
from utils.commons import log_to_disk, get_backbone
from utils.datasets import get_split, resample, MTLDataset
from config.task_config import MTLConfig
from multitask.probe import MultiTaskProbe
from gradnorm_pytorch import GradNormLossWeighter

BACKBONE_LR_RATIO = 0.1

class Trainer:
    """
    Encapsulates the training, validation, and testing logic for a multi-task probe.
    
    This class is responsible for:
    - Setting up the model, optimizer, and data loaders.
    - Running the training and evaluation loops.
    - Logging results and saving checkpoints.
    """
    def __init__(self, config: MTLConfig, args: argparse.Namespace):
        self.config = config
        self.args = args
        self.device = torch.device(config.device)
        self.scaler = torch.amp.GradScaler('cuda')
        self.version_name = 'Siglip2' if 'google' in args.version else args.version
        self.saving_interval = 15
        self.start_epoch = 0
        self.unfrozen_layers = 0 # besides attention pooling
        self.best_path=None
        self.ckpt_folder = 'ckpt'
        self.scheduler_type = 'lr_on_platue' # or cosine
        self.probing_type = (
            'k-probe' if self.args.k_probes
            else 'mhca-moe' if self.args.moe
            else 'classic'
        )
        # lower lr for backboens
        self.lr_config = {
            'head': self.args.learning_rate,         
            'backbone': self.args.learning_rate * BACKBONE_LR_RATIO 
        }
        # at epoch key unfreeze the first value layers
        self.unfreeze_schedule = {
            1 : 2, 
            2 : 2, 
            3 : 4,
            4 : 5
        }

        self.use_uncertainty_weighting = config.use_uncertainty_weighting

        # dynamic unfreezing
        self.unfreeze_on_plateau = True  
        self.unfreeze_step = 1           
        self.unfreeze_patience = 2       
        self.max_unfrozen_layers = 4     

        # State trackers for the dynamic strategy
        self.plateau_counter = 0
        self.best_accuracy_in_stage = -float('inf')

        # grad norm parameters
        self.use_grad_norm = config.use_grad_norm 
        self.grad_norm_alpha = config.grad_norm_alpha 
        self.initial_task_losses = None

        # These will be initialized in self.setup()
        self.mtl_probe = None
        self.transform = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.optimizer = None
        self.scheduler = None
        self.criterions = {}
        if args.testing:
            self._setup_model_and_optimizer()
            self._setup_criterions({})
        else:
            self.setup()
        
    def setup(self):
        """Initializes all components required for training."""
        print("\n--- Setting up trainer ---")
        self._setup_model_and_optimizer()
        class_weights = self._setup_loaders()
        self._setup_criterions(class_weights)


    def _setup_model_and_optimizer(self):
        """Initializes the model, optimizer, and scheduler."""
        print("[TRAINER] Setting up model and optimizer...")
        backbone, self.transform, hidden_size = get_backbone(self.args.version, self.args.ckpt_path)

        task_dims = {task.name.lower(): task.num_classes for task in self.config.tasks}
        
        self.mtl_probe = MultiTaskProbe(
            backbone=backbone,
            backbone_output_dim=hidden_size,
            tasks=task_dims,
            use_moe=self.args.moe,
            use_k_probes=self.args.k_probes
        ).to(self.device)

        # self.mtl_probe = torch.compile(self.mtl_probe)

        # grad norm requires that at least one other layer is unfrozen
        initial_unfrozen_layers = 1 if self.use_grad_norm else 0
        initial_param_groups = self.mtl_probe.get_parameter_groups(initial_unfrozen_layers, 
                                                                using_uncertainty=self.use_uncertainty_weighting,
                                                                using_grad_norm=self.use_grad_norm)
        self.unfrozen_layers = initial_unfrozen_layers

        optimizer_param_groups = []
        for group in initial_param_groups:
            group_name = group['name']
            lr = self.lr_config.get(group_name, self.args.learning_rate)
            optimizer_param_groups.append({'params': group['params'], 'lr': lr})

        self.optimizer = optim.AdamW(optimizer_param_groups, lr=self.args.learning_rate, fused=True)
        
        if self.use_grad_norm:
            self.loss_weighter = GradNormLossWeighter(
                num_losses = 3,
                learning_rate = 1e-4,
                restoring_force_alpha = 0.,                
                grad_norm_parameters =  None
            )


        if self.scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2)
        else:
            self.scheduler = ReduceLROnPlateau(self.optimizer, factor = 0.5) # halve the learning rate on stagnation


        if self.args.resume_from_ckpt and os.path.exists(self.args.resume_from_ckpt):
            print(f"[TRAINER] Resuming from checkpoint: {self.args.resume_from_ckpt}")
            self.start_epoch = self.mtl_probe.load(self.args.resume_from_ckpt, optimizer=self.optimizer, scheduler=self.scheduler)
        else:
            print("[TRAINER] Starting from scratch.")
    
    def unfreeze_layers(self, num_layers_to_unfreeze : int):
        """Unfreeze more layers of the backbone """
        new_param_groups = self.mtl_probe.unfreeze_and_get_new_params(num_layers_to_unfreeze)

        if not new_param_groups:
            print("[Optimizer] No new layers were unfrozen.")
            return

        # Add the new parameter groups to the existing optimizer
        for group in new_param_groups:
            group_name = group['name']
            lr = self.lr_config.get(group_name, self.args.learning_rate * BACKBONE_LR_RATIO) # Default to backbone LR
            
            param_group_dict = {'params': group['params'], 'lr': lr}
            
            print(f"[Optimizer] Adding new parameter group '{group_name}' with LR: {lr}")
            self.optimizer.add_param_group(param_group_dict)

        print("[Optimizer] Successfully added new parameters to the existing optimizer.")

        if isinstance(self.scheduler, ReduceLROnPlateau):
            print("[Scheduler] Re-initializing ReduceLROnPlateau scheduler.")
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, 
                factor=0.5, 
            )
            


    def _setup_loaders(self) -> Dict[str, torch.Tensor]:
        """Initializes the dataloaders and returns class weights."""
        print("[TRAINER] Setting up data loaders...")
        dataset_train = MTLDataset(
            csv_path=self.config.train_csv,
            transform=self.transform,
            augment=True,
            root_dir=self.config.dataset_root,
            balance=True
        )
        
        dataset_val = MTLDataset(
            csv_path=self.config.val_csv,
            transform=self.transform,
            augment=False,
            root_dir=self.config.dataset_root,
            balance=False
        )

        self.train_loader = DataLoader(dataset_train, batch_size=self.args.batch_size, num_workers=self.config.num_workers, pin_memory=True, shuffle=True)
        if self.config.num_workers <= 2:
            print(f'[WARNING] THE NUMBER OF WORKERS IS ONLY {self.config.num_workers} THIS MAY RESULT IN A SLOW TRAINING')
        self.val_loader = DataLoader(dataset_val, batch_size=self.args.batch_size, num_workers=self.config.num_workers, pin_memory=True, shuffle=False)
        
        return dataset_train.get_inverse_weights_loss()


    def _setup_test_loader(self, save_extreme_wrong=False):
        """Initializes the test dataloader when needed."""
        print("[TRAINER] Setting up test data loader...")
        dataset_test = MTLDataset(
            csv_path=self.config.test_csv,
            transform=self.transform,
            augment=False,
            root_dir=self.config.dataset_root,
            balance=False,
            return_path=save_extreme_wrong
        )

        self.test_loader = DataLoader(dataset_test, batch_size=self.args.batch_size, num_workers=self.config.num_workers, pin_memory=True, shuffle=False)

    def _setup_criterions(self, class_weights: Dict[str, torch.Tensor]):
        """Initializes loss functions with optional class weighting."""
        print("[TRAINER] Setting up criterions...")
        for task in self.config.tasks:
            params = {'ignore_index': self.config.ignore_index}
            if task.use_weighted_loss and task.name in class_weights:
                params['weight'] = class_weights[task.name].to(self.device)
                print(f"[TRAINER] Using weighted loss for task: {task.name}")
            self.criterions[task.name] = task.criterion(**params)


    def eval_unfreeze_dynamic(self, val_accuracy):
        """Evaluates if it's necessary to unfreeze more layers of the backbone, and does so if needed """
        
        if val_accuracy > self.best_accuracy_in_stage:
            print(f"[Plateau Check] Validation accuracy improved to {val_accuracy:.4f}. Resetting counter.")
            self.best_accuracy_in_stage = val_accuracy
            self.plateau_counter = 0
        else:
            self.plateau_counter += 1
            print(f"[Plateau Check] No improvement. Plateau counter is now {self.plateau_counter}/{self.unfreeze_patience}.")
                    
            if self.plateau_counter >= self.unfreeze_patience:
                if self.unfrozen_layers < self.max_unfrozen_layers:
                    print(f"[ACTION] Validation accuracy plateaued for {self.unfreeze_patience} epochs. Unfreezing {self.unfreeze_step} more layers.")
                    
                    self.unfreeze_layers(self.unfrozen_layers + self.unfreeze_step) 
                    self.unfrozen_layers += self.unfreeze_step
                    self.plateau_counter = 0
                    self.best_accuracy_in_stage = -float('inf')
                    print(f"[Dynamic Unfreezing] Now {self.unfrozen_layers} of the backbone are unfrozen.")
                else:
                    print("[Plateau Check] Max unfrozen layers reached. No more unfreezing.")
                    self.unfreeze_on_plateau = False 


    def train(self):
        """Runs the main training loop."""
        print("\n--- Starting Training ---")
        self.config.output_folder.mkdir(parents=True, exist_ok=True)
        (self.config.output_folder / 'ckpt').mkdir(exist_ok=True)
        (self.config.output_folder / 'confusion_matrices').mkdir(exist_ok=True)
        (self.config.output_folder / 'confusion_matrices_normalized').mkdir(exist_ok=True)
        maximum_val_accuracy = -float('inf')
        maximum_val_age = -float('inf')

        for epoch in range(self.start_epoch, self.args.epochs):
            
            if epoch in self.unfreeze_schedule and self.unfreeze_on_plateau == False:
                print(f'[TRAINER] UNFREEZE OF LAYER {self.unfreeze_schedule[epoch]}')
                self.unfreeze_layers(self.unfreeze_schedule[epoch])
                self.unfrozen_layers = self.unfreeze_schedule[epoch]


            print(f"\nEpoch {epoch + 1}/{self.args.epochs}")
            
            train_results = self._run_epoch(self.train_loader, is_training=True, epoch_idx=epoch)
            print(f"TRAINING Avg Loss: {train_results['avg_loss']:.4f}, Task Losses: {train_results['task_losses']}")
            
            val_results = self._run_epoch(self.val_loader, is_training=False, description="Validating")
            print(f"VALIDATION Avg Loss: {val_results['avg_loss']:.4f}, Task Accuracies: {val_results['accuracies']}")
            
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_results['avg_loss'])

            for task_name, cm in val_results['confusion_matrices'].items():
                self.save_confusion_matrix(cm, task_name, epoch+1)
            
            for task_name, cm in val_results['confusion_matrices_normalized'].items():
                self.save_confusion_matrix(cm, task_name, epoch+1, normalized=True)

            self._log_epoch_results(epoch, train_results, val_results)

            
            val_accuracy = val_results['avg_accuracy']
            val_age = val_results['task_losses']['Age']

            # dynamic un-freezing logic 
            if self.unfreeze_on_plateau:
                self.eval_unfreeze_dynamic(val_accuracy)

            # saving logic (save each saving_interval, or save when avg_val_accuracy has reached a new height or save if its the last epoch)
            if (epoch + 1) % self.saving_interval == 0 or val_accuracy > maximum_val_accuracy or (epoch + 1) == self.args.epochs:
                if val_accuracy > maximum_val_accuracy:
                    print(f"[TRAINER] New best validation accuracy: {val_accuracy:.5f}. Saving model.")
                    maximum_val_accuracy = val_accuracy
                    if self.best_path:  
                        if os.path.exists(self.best_path):
                            os.remove(self.best_path)
                    self.best_path = self.config.output_folder / 'ckpt' / f"mtl_{self.version_name}_{epoch + 1}.pt"
                
                save_path = self.config.output_folder / 'ckpt' / f"mtl_{self.version_name}_ul{self.unfrozen_layers}_{epoch + 1}.pt"
                self.mtl_probe.save(path=str(save_path), epoch=epoch + 1, optimizer=self.optimizer, scheduler=self.scheduler)

    def test(self, ckpt_path=None):
        """Evaluates the model on the test set. Saves confusion matrices and generates a test_result.txt"""
        self._setup_test_loader(save_extreme_wrong=False)

        if ckpt_path:
            self.mtl_probe.load(ckpt_path)
        else:
            self.mtl_probe.load(self.best_path)

        print("\n--- Starting Testing ---")
        test_results = self._run_epoch(self.test_loader, is_training=False, description="Testing",save_extreme_wrong=False)
        
        for task_name, cm in test_results['confusion_matrices'].items():
            self.save_confusion_matrix(cm, task_name, 0, is_final=True, accuracy=test_results['avg_accuracy'])
        
        with open("./test_results.txt", "w") as f:
            f.write("\n--- Test Results ---\n")
            f.write(f"Average Loss: {test_results['avg_loss']:.4f}\n")
            f.write(f"Average Accuracy: {test_results['avg_accuracy']:.2f}%\n")
            
            f.write("\nPer-Task Losses:\n")
            for name, loss in test_results['task_losses'].items():
                f.write(f"  - {name}: {loss:.4f}\n")
            
            f.write("\nPer-Task Accuracies:\n")
            for name, acc in test_results['accuracies'].items():
                f.write(f"  - {name}: {acc:.2f}%\n")
            
            for name, acc in test_results['top_k_accuracies'].items():
                f.write(f"  - {name}: {acc:.2f}% @2\n")


    def _run_epoch(self, loader: DataLoader, is_training: bool, description: str = "Training", epoch_idx: int = 0, k: int = 2, save_extreme_wrong :bool =False) -> Dict[str, Any]:
        """
        Runs a single epoch of training or evaluation for a multi-task model.
        
        Args:
            loader: The DataLoader for the current dataset.
            is_training: If True, performs training (backpropagation), otherwise evaluation.
            description: A string for the progress bar.
            epoch_idx: The current epoch index, used for the learning rate scheduler.
            k: The value for top-k accuracy calculation.
            
        Returns:
            A dictionary containing metrics for the epoch, including top-k accuracy.
        """
        self.mtl_probe.train(is_training)
        
        total_loss = 0.0
        total_bal_loss = 0.0
        task_loss_totals = {task.name: 0.0 for task in self.config.tasks}
        task_sample_counts = {task.name: 0 for task in self.config.tasks}
        
        # Placeholders for predictions and true labels, used only in evaluation
        all_preds = {task.name: [] for task in self.config.tasks}
        all_true = {task.name: [] for task in self.config.tasks}
        
        correct_top_k_counts = {task.name: 0 for task in self.config.tasks}

        pbar = tqdm(enumerate(loader), total=len(loader), desc=description)

        context = torch.no_grad() if not is_training else torch.enable_grad()
        # --------------- EPOCH --------------- #
        with context:
            for i, batch in pbar:
                if save_extreme_wrong:
                    images, labels, paths = batch
                else:
                    images, labels = batch
                # grad norm is not compatible with autocasting, has they both "mess" with gradients 
                # not self.use_grad_norm
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled= not self.use_grad_norm):
                    loss, task_losses, task_outputs, unweighted_task_losses, bal_loss, losses  = self._process_batch(images, labels, self.use_grad_norm)
                
                
                if is_training:
                    self.optimizer.zero_grad()

                    if self.use_grad_norm:
                        loss.backward(retain_graph=True)
                        if self.initial_task_losses is None:
                            self.initial_task_losses = [l.item() for l in unweighted_task_losses]
                        
                        last_shared_layer = self.mtl_probe.get_last_shared_layer()
                        grad_norms = []

                        for task_loss in unweighted_task_losses:
                            grads = torch.autograd.grad(task_loss, last_shared_layer.parameters(), retain_graph=True, allow_unused=True)
                            valid_grads = [g.view(-1) for g in grads if g is not None]
                            if valid_grads:
                                grad_norm = torch.norm(torch.cat(valid_grads)) # L2 norm as described in paper
                                grad_norms.append(grad_norm)

                        if not grad_norms: 
                            self.optimizer.step()
                            continue

                        
                        # Convert grad_norms list to a tensor
                        stacked_grad_norms = torch.stack(grad_norms)

                        # Calculate the weighted gradient norms. 
                        # The GradNorm paper's loss is based on w_i * G_W(L_i)
                        weighted_grad_norms = self.mtl_probe.loss_weights * stacked_grad_norms
                        avg_grad_norm = torch.mean(weighted_grad_norms).detach()

                        loss_ratios = [l.item() / initial_l for l, initial_l in zip(unweighted_task_losses, self.initial_task_losses)]
                        avg_loss_ratio = sum(loss_ratios) / len(loss_ratios)
                        relative_inverse_rates = [l / avg_loss_ratio for l in loss_ratios] # r_i (t) in the paper
                        
                        # Calculate the target gradient norms for each task
                        target_grad_norms = torch.tensor([avg_grad_norm * (r ** self.grad_norm_alpha) for r in relative_inverse_rates], device=self.device)

                        # The grad_norm_loss is the L1 distance between the current weighted norms and the targets.
                        grad_norm_loss = torch.sum(torch.abs(weighted_grad_norms - target_grad_norms))


                        self.mtl_probe.loss_weights.grad = torch.autograd.grad(grad_norm_loss, self.mtl_probe.loss_weights)[0]
                        print(self.mtl_probe.loss_weights.data)
                        self.optimizer.step()

                        with torch.no_grad():
                            # normalization of the weights so they sum up to the number of tasks 
                            self.mtl_probe.loss_weights.data = (self.mtl_probe.loss_weights.data / self.mtl_probe.loss_weights.data.sum()) * self.mtl_probe.num_tasks   
                    else:

                        # self.loss_weighter.backward(losses, shared_features)
                        # print(self.loss_weighter.loss_weights)
                        # self.optimizer.step()
                        # self.optimizer.zero_grad()
                        self.scaler.scale(loss).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()

                    if isinstance(self.scheduler, CosineAnnealingWarmRestarts):
                        self.scheduler.step(epoch_idx + i / len(loader))

                total_loss += loss.item()
                total_bal_loss += bal_loss.item()

                task_loss_str = ", ".join([f"{name}: {l:.4f}" for name, l in task_losses.items()])
                pbar.set_description(f"{description} - Avg Loss: {total_loss / (i + 1):.4f} | Batch Losses: [{task_loss_str}]")
                if is_training and self.scheduler:
                    pbar.set_postfix(lr=f"{self.scheduler.get_last_lr()[0]:.6f}")
                
                # Accumulate losses, predictions, and top-k counts
                for idx, task in enumerate(self.config.tasks):
                    target = labels[:, idx]
                    valid_mask = (target != self.config.ignore_index)
                    if not valid_mask.any():
                        continue

                    num_valid = valid_mask.sum().item()
                    task_sample_counts[task.name] += num_valid
                    task_loss_totals[task.name] += task_losses.get(task.name, 0.0) * num_valid
                    
                    if not is_training:
                        valid_outputs = task_outputs[idx][valid_mask]
                        valid_targets = target[valid_mask].to(self.device)


                        preds = valid_outputs.argmax(dim=1)
                        all_preds[task.name].extend(preds.cpu().numpy())
                        all_true[task.name].extend(valid_targets.cpu().numpy())


                        valid_targets_gpu = valid_targets.to(valid_outputs.device)
                        
                        _, top_k_preds = torch.topk(valid_outputs, k, dim=1)
                        expanded_labels = valid_targets_gpu.view(-1, 1).expand_as(top_k_preds)
                        
                        correct_in_batch = torch.sum(torch.any(top_k_preds == expanded_labels, dim=1)).item()
                        correct_top_k_counts[task.name] += correct_in_batch

                        if save_extreme_wrong and (task.name == 'age' or task.name == 'Age'):
                            misclassified = preds != valid_targets

                            valid_paths = [p for i, p in enumerate(paths) if valid_mask[i]]

                            for idx in torch.where(misclassified)[0]:
                                true_label = valid_targets[idx].item()
                                pred_label = preds[idx].item()

                                if abs(true_label - pred_label) > 3:
                                    output_dir = self.config.output_folder / f"extreme_wrong_classifications/true_{true_label}/pred_{pred_label}"
                                    output_dir.mkdir(parents=True, exist_ok=True)
                                    original_image_tensor = images[valid_mask][idx]
                                    img_path = Path(valid_paths[idx])
                                    save_image(original_image_tensor, output_dir / f"{img_path.stem}.png")
                # break
                
            # --------------- EPOCH --------------- #
                  

        # Compile results
        results = {}
        results['avg_loss'] = total_loss / len(loader)
        results['task_losses'] = {name: total / task_sample_counts[name] if task_sample_counts[name] > 0 else 0 for name, total in task_loss_totals.items()}
        
        if not is_training:
            results['bal_loss'] = total_bal_loss / len(loader)
            results['accuracies'] = {name: accuracy_score(all_true[name], all_preds[name]) * 100 for name in all_true if all_true[name]}
            results['avg_accuracy'] = sum(results['accuracies'].values()) / len(results['accuracies']) if results['accuracies'] else 0
            results['confusion_matrices'] = {
                name: confusion_matrix(all_true[name], all_preds[name], labels=range(self.config.task_map[name].num_classes))
                for name in all_true if all_true[name]
            }
            results['confusion_matrices_normalized']  = {
                name: confusion_matrix(all_true[name], all_preds[name], labels=range(self.config.task_map[name].num_classes),normalize='true')
                for name in all_true if all_true[name]
            }
            results['top_k_accuracies'] = {
                name: (correct_top_k_counts[name] / task_sample_counts[name]) * 100
                for name in correct_top_k_counts if task_sample_counts[name] > 0
            }
            results['avg_top_k_accuracy'] = sum(results['top_k_accuracies'].values()) / len(results['top_k_accuracies']) if results['top_k_accuracies'] else 0

        return results
    
    def _process_batch(self, images: torch.Tensor, labels: torch.Tensor, use_grad_norm : bool) -> Tuple[torch.Tensor, Dict[str, float], List[torch.Tensor]]:
        """Processes a single batch of data, returning total loss weighted, unweighted task losses as a dictionary, model outputs and unweighted task_losses as a list"""
        images, labels = images.to(self.device), labels.to(self.device)
        
        # Forward pass
        if self.args.moe:
            task_outputs, bal_loss, _ = self.mtl_probe(images)
        else:
            task_outputs, bal_loss = self.mtl_probe(images), 0

        
        total_loss = 0.0
        task_losses = {}
        losses = []
        unweighted_task_losses = []

        for idx, task in enumerate(self.config.tasks):
            criterion = self.criterions[task.name]
            task_loss = criterion(task_outputs[idx], labels[:, idx])
            losses.append(task_loss)
            # to avoid to add nan to the batch loss
            if torch.isfinite(task_loss):
                unweighted_task_losses.append(task_loss)
                if self.config.use_uncertainty_weighting:
                    weighted_loss = 0.5 * torch.exp(-self.mtl_probe.log_var[idx]) * task_loss + 0.5 * self.mtl_probe.log_var[idx]
                    total_loss += weighted_loss
                elif self.config.use_grad_norm:
                    total_loss += self.mtl_probe.loss_weights[idx] * task_loss
                else: # default static weight
                    total_loss += task_loss * task.weight 

                task_losses[task.name] = task_loss.item()


        # Add balancing loss for MoE models
        if self.args.moe:
            total_loss += 0.05 * bal_loss
        return total_loss, task_losses, task_outputs, unweighted_task_losses, bal_loss, losses

    def _log_epoch_results(self, epoch_idx: int, train_results: Dict, val_results: Dict):
        """Logs the results of an epoch to a CSV file."""
        lr = self.scheduler.get_last_lr()[0] if self.scheduler else self.args.learning_rate
        loss_weights_str = ""

        if self.use_grad_norm:
            weights_values = self.mtl_probe.loss_weights.detach().cpu().tolist()
            loss_weights_str = ",".join([f"{w:.4f}" for w in weights_values])
        if self.use_uncertainty_weighting:
            weights_values = self.mtl_probe.log_var.detach().cpu().tolist()
            loss_weights_str = ",".join([f"{w:.4f}" for w in weights_values])

        if train_results:
            log_items = [
                epoch_idx + 1,
                f"{train_results['avg_loss']:.4f}",
                *[f"{loss:.4f}" for loss in train_results['task_losses'].values()],
                f"{val_results['avg_loss']:.4f}",
                *[f"{loss:.4f}" for loss in val_results['task_losses'].values()],
                f"{val_results['avg_accuracy']:.3f}",
                *[f"{acc:.3f}" for acc in val_results['accuracies'].values()],
                f"{lr:.6f}", f"{self.unfrozen_layers}",
                loss_weights_str,
                f"{val_results['bal_loss']:.4f}"
            ]
        else:
            log_items = [
                epoch_idx + 1,
                f"{val_results['avg_loss']:.4f}",
                *[f"{loss:.4f}" for loss in val_results['task_losses'].values()],
                f"{val_results['avg_accuracy']:.3f}",
                *[f"{acc:.3f}" for acc in val_results['top_k_accuracies'].values()],
                *[f"{acc:.3f}" for acc in val_results['accuracies'].values()],
                f"{lr:.6f}"
            ]

        log_string = ",".join(map(str, log_items))
        
        log_to_disk(
            self.config.output_folder,
            log_string,
            f'mtl_{self.probing_type}_{self.version_name}' if train_results else f'mtl_{self.probing_type}_{self.version_name}_testing',
            header=self.config.header
        )

    def save_confusion_matrix(self, cm: np.ndarray, task_name: str, epoch:int,  is_final: bool = False, accuracy=None, normalized=False):
        """Saves a confusion matrix plot to disk."""
        task = self.config.task_map[task_name]
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=task.class_labels)
        disp.plot()
        if normalized:
            disp.plot(values_format='.2f')
        
        suffix = '_test_set' if is_final else str(epoch)
        if accuracy:
            suffix = f'{suffix}_{accuracy:.2f}'
        cm_path = self.config.output_folder / 'confusion_matrices' /f'cm_mtl_{self.probing_type}_{self.version_name}_{task_name}{suffix}.jpg'
        if normalized:
            cm_path = self.config.output_folder / 'confusion_matrices_normalized' /f'cm_mtl_{self.probing_type}_{self.version_name}_{task_name}{suffix}.jpg'
        plt.savefig(cm_path, bbox_inches='tight')
        plt.close() # Close the plot to free memory
        print(f"Confusion matrix for '{task_name}' saved to {cm_path}")

    def load_heads(self, ckpt_paths):
        print('[TRAINER] Loading pre-trained heads...')
        self.mtl_probe.load_heads(ckpt_paths)
        self.probing_type = self.probing_type + '_pre-trained-heads_'
        print('[TRAINER] Loaded pre-trained heads.')

    @staticmethod
    def cleanup():
        """Attempts to gracefully terminate lingering DataLoader worker processes."""
        print("\n[Trainer Cleanup] Searching for active worker processes...")
        import multiprocessing
        active_procs = multiprocessing.active_children()
        
        if not active_procs:
            print("[Trainer Cleanup] No active child processes found.")
            return

        for process in active_procs:
            print(f"[Trainer Cleanup] Shutting down process: {process.name} (PID: {process.pid})")
            process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                print(f"[Trainer Cleanup] Process {process.pid} did not terminate, forcing kill.")
                process.kill()
        
        print("[Trainer Cleanup] All worker processes have been handled.")



