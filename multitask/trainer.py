import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any,  Dict, List, Tuple, Type
import uuid

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
# from multitask.multitask_probe import MultiTaskProbe
from multitask.probe import MultiTaskProbe




# TODO Gradual unfreezing
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
        self.saving_interval = 10
        self.start_epoch = 0
        self.unfrozen_layers = 0 # besides attention pooling
        self.best_path=None
        self.ckpt_folder = 'ckpt'
        self.scheduler_type = 'lr_on_platue' # or cosine
        # classic, k-probe, mhca-moe,
        self.probing_type = (
            'k-probe' if self.args.k_probes
            else 'mhca-moe' if self.args.moe
            else 'classic'
        )
        # lower lr for backboens
        self.lr_config = {
            'head': self.args.learning_rate,         
            'backbone': self.args.learning_rate * 0.1  
        }
        # at epoch key unfreeze the first value layers
        self.unfreeze_schedule = {
            2 : 2, 
            4 : 2, 
            6 : 3,
            8 : 4
        }

        # These will be initialized in self.setup()
        self.mtl_probe = None
        self.transform = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.optimizer = None
        self.scheduler = None
        self.criterions = {}
        self.setup()
        
    def setup(self):
        """Initializes all components required for training."""
        print("\n--- Setting up trainer ---")
        self._setup_model_and_optimizer()
        class_weights = self._setup_loaders()
        self._setup_criterions(class_weights)

    def unfreeze_layers(self, num_layers_to_unfreeze:int = None, k=1):
        if num_layers_to_unfreeze:
            named_groups = self.mtl_probe.unfreeze_layers(num_layers_to_unfreeze)
            self.unfrozen_layers = num_layers_to_unfreeze
        else:
            named_groups = self.mtl_probe.unfreeze_layers(self.unfrozen_layers + k)
            self.unfrozen_layers += k
        
        optimizer_param_groups = []
        for group in named_groups:
            group_name = group['name']
            lr = self.lr_config[group_name]
            optimizer_param_groups.append({
                'params': group['params'],
                'lr': lr
            })
        self.optimizer = optim.AdamW(optimizer_param_groups, fused=True)

        if self.scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2)
        else:
            self.scheduler = ReduceLROnPlateau(self.optimizer, factor = 0.5) # half learning rate on stagnation

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
        self.mtl_probe = torch.compile(self.mtl_probe)
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.mtl_probe.parameters()),
            lr=self.args.learning_rate,
            fused=True
        )
        if self.scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2)
        else:
            self.scheduler = ReduceLROnPlateau(self.optimizer, factor = 0.5) # half learning rate on stagnation


        if self.args.resume_from_ckpt and os.path.exists(self.args.resume_from_ckpt):
            print(f"[TRAINER] Resuming from checkpoint: {self.args.resume_from_ckpt}")
            self.start_epoch = self.mtl_probe.load(self.args.resume_from_ckpt, optimizer=self.optimizer, scheduler=self.scheduler)
        else:
            print("[TRAINER] Starting from scratch.")

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
        self.val_loader = DataLoader(dataset_val, batch_size=self.args.batch_size, num_workers=self.config.num_workers, pin_memory=True)
        
        return dataset_train.get_inverse_weights_loss()


    def _setup_test_loader(self):
        """Initializes the test dataloader when needed."""
        print("[TRAINER] Setting up test data loader...")
        dataset_test = MTLDataset(
            csv_path=self.config.test_csv,
            transform=self.transform,
            augment=False,
            root_dir=self.config.dataset_root,
            balance=False,
            return_path=True
        )
        self.test_loader = DataLoader(dataset_test, batch_size=self.args.batch_size, num_workers=self.config.num_workers, pin_memory=True)

    def _setup_criterions(self, class_weights: Dict[str, torch.Tensor]):
        """Initializes loss functions with optional class weighting."""
        print("[TRAINER] Setting up criterions...")
        for task in self.config.tasks:
            params = {'ignore_index': self.config.ignore_index}
            if task.use_weighted_loss and task.name in class_weights:
                params['weight'] = class_weights[task.name].to(self.device)
                print(f"[TRAINER] Using weighted loss for task: {task.name}")
            self.criterions[task.name] = task.criterion(**params)
        

    def _get_save_path(self, epoch: int, is_head_only: bool = False) -> Path:
        """Generates a consistent path for saving checkpoints."""
        task_name = f'mtl_{self.unfrozen_layers}'
        suffix = f"head+{str(epoch)}" if is_head_only else str(epoch)
        filename = f"{task_name}_{self.version_name}_{suffix}.pt"
        return self.config.output_folder / 'ckpt' / filename
    

    def train(self):
        """Runs the main training loop."""
        print("\n--- Starting Training ---")
        self.config.output_folder.mkdir(parents=True, exist_ok=True)
        (self.config.output_folder / 'ckpt').mkdir(exist_ok=True)
        
        maximum_val_accuracy = -float('inf')

        for epoch in range(self.start_epoch, self.args.epochs):
            
            if epoch in self.unfreeze_schedule:
                print(f'[TRAINER] UNFREEZE OF LAYER {self.unfreeze_schedule[epoch]}')
                self.unfreeze_layers(self.unfreeze_schedule[epoch])


            print(f"\nEpoch {epoch + 1}/{self.args.epochs}")
            
            train_results = self._run_epoch(self.train_loader, is_training=True, epoch_idx=epoch)
            print(f"TRAINING Avg Loss: {train_results['avg_loss']:.4f}, Task Losses: {train_results['task_losses']}")
            
            val_results = self._run_epoch(self.val_loader, is_training=False, description="Validating")
            print(f"VALIDATION Avg Loss: {val_results['avg_loss']:.4f}, Task Accuracies: {val_results['accuracies']}")
            
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_results['avg_loss'])

            for task_name, cm in val_results['confusion_matrices'].items():
                self.save_confusion_matrix(cm, task_name, epoch+1)
            
            self._log_epoch_results(epoch, train_results, val_results)
            
            val_accuracy = val_results['avg_accuracy']
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
        """Evaluates the model on the test set."""
        self._setup_test_loader()

        if ckpt_path:
            self.mtl_probe.load(ckpt_path)
        else:
            self.mtl_probe.load(self.best_path)

        print("\n--- Starting Testing ---")
        test_results = self._run_epoch(self.test_loader, is_training=False, description="Testing",save_extreme_wrong=True)
        
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
        task_loss_totals = {task.name: 0.0 for task in self.config.tasks}
        task_sample_counts = {task.name: 0 for task in self.config.tasks}
        
        # Placeholders for predictions and true labels, used only in evaluation
        all_preds = {task.name: [] for task in self.config.tasks}
        all_true = {task.name: [] for task in self.config.tasks}
        
        correct_top_k_counts = {task.name: 0 for task in self.config.tasks}

        pbar = tqdm(enumerate(loader), total=len(loader), desc=description)

        context = torch.no_grad() if not is_training else torch.enable_grad()
        with context:
            for i, batch in pbar:
                if save_extreme_wrong:
                    images, labels, paths = batch
                else:
                    images, labels = batch
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    loss, task_losses, task_outputs = self._process_batch(images, labels)
                
                if is_training:
                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    #loss.backward()
                    #self.optimizer.step()
                    if isinstance(self.scheduler, CosineAnnealingWarmRestarts):
                        self.scheduler.step(epoch_idx + i / len(loader))

                total_loss += loss.item()
                
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
                                    #print('trovato uno')
                                    original_image_tensor = images[valid_mask][idx]
                                    img_path = Path(valid_paths[idx])
                                    save_image(original_image_tensor, output_dir / f"{img_path.stem}.png")
                   
        # Compile results
        results = {}
        results['avg_loss'] = total_loss / len(loader)
        results['task_losses'] = {name: total / task_sample_counts[name] if task_sample_counts[name] > 0 else 0 for name, total in task_loss_totals.items()}
        
        if not is_training:
            results['accuracies'] = {name: accuracy_score(all_true[name], all_preds[name]) * 100 for name in all_true if all_true[name]}
            results['avg_accuracy'] = sum(results['accuracies'].values()) / len(results['accuracies']) if results['accuracies'] else 0
            results['confusion_matrices'] = {
                name: confusion_matrix(all_true[name], all_preds[name], labels=range(self.config.task_map[name].num_classes))
                for name in all_true if all_true[name]
            }
            results['top_k_accuracies'] = {
                name: (correct_top_k_counts[name] / task_sample_counts[name]) * 100
                for name in correct_top_k_counts if task_sample_counts[name] > 0
            }
            results['avg_top_k_accuracy'] = sum(results['top_k_accuracies'].values()) / len(results['top_k_accuracies']) if results['top_k_accuracies'] else 0

        return results
    
    def _process_batch(self, images: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float], List[torch.Tensor]]:
        """Processes a single batch of data, returning total loss, task losses, and outputs."""
        images, labels = images.to(self.device), labels.to(self.device)
        
        # Forward pass
        if self.args.moe:
            task_outputs, bal_loss, _ = self.mtl_probe(images)
        else:
            task_outputs, bal_loss = self.mtl_probe(images), 0

        total_loss = 0.0
        task_losses = {}
        
        for idx, task in enumerate(self.config.tasks):
            criterion = self.criterions[task.name]
            task_loss = criterion(task_outputs[idx], labels[:, idx])
            
            # Store individual loss value, accounting for cases where no valid samples exist
            # to avoid to add nan to the batch loss
            if torch.isfinite(task_loss):
                if self.config.use_uncertainty_weighting:
                    weighted_loss = 0.5 * torch.exp(-self.mtl_probe.log_var[idx]) * task_loss + 0.5 * self.mtl_probe.log_var[idx]
                    total_loss += weighted_loss
                elif self.config.use_grad_norm:
                    raise NotImplementedError("Grad norm has yet to be implemented")
                else: # default static weight
                    total_loss += task_loss * task.weight 

                task_losses[task.name] = task_loss.item()


        # Add balancing loss for MoE models
        if self.args.moe:
            total_loss += 0.05 * bal_loss
        return total_loss, task_losses, task_outputs

    def _log_epoch_results(self, epoch_idx: int, train_results: Dict, val_results: Dict):
        """Logs the results of an epoch to a CSV file."""
        lr = self.scheduler.get_last_lr()[0] if self.scheduler else self.args.learning_rate
        
        if train_results:
            log_items = [
                epoch_idx + 1,
                f"{train_results['avg_loss']:.4f}",
                *[f"{loss:.4f}" for loss in train_results['task_losses'].values()],
                f"{val_results['avg_loss']:.4f}",
                *[f"{loss:.4f}" for loss in val_results['task_losses'].values()],
                f"{val_results['avg_accuracy']:.3f}",
                *[f"{acc:.3f}" for acc in val_results['accuracies'].values()],
                f"{lr:.6f}"
            ]
        else:
            log_items = [
                epoch_idx + 1,
                f"{val_results['avg_loss']:.4f}",
                *[f"{loss:.4f}" for loss in val_results['task_losses'].values()],
                f"{val_results['avg_accuracy']:.3f}",
                *[f"{acc:.3f}" for acc in val_results['top_k_accuracies'].values()]
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

    def save_confusion_matrix(self, cm: np.ndarray, task_name: str, epoch:int,  is_final: bool = False, accuracy=None):
        """Saves a confusion matrix plot to disk."""
        task = self.config.task_map[task_name]
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=task.class_labels)
        disp.plot()
        
        suffix = '_test_set' if is_final else str(epoch)
        if accuracy:
            suffix = f'{suffix}_{accuracy:.2f}'
        cm_path = self.config.output_folder / 'confusion_matricies' /f'cm_mtl_{self.probing_type}_{self.version_name}_{task_name}{suffix}.jpg'
        
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




if __name__ == '__main__':
    from types import SimpleNamespace
    import tempfile
    def test_trainer_save_and_load():
        """A standalone function to test the Trainer's save/load cycle."""
        # Use a temporary directory that gets cleaned up automatically
        with tempfile.TemporaryDirectory() as tempdir:
            temp_path = Path(tempdir)
            checkpoint_path = temp_path / "test_ckpt.pt"

            # 1. ARRANGE (Setup First Trainer)
            # ------------------------------------
            print("--- Setting up Trainer 1 for saving ---")
            
            # Mock args and config
            args1 = SimpleNamespace(
                version='google/Siglip2-base-patch16-224',
                ckpt_path=None,
                learning_rate=0.01,
                resume_from_ckpt=None, # Not resuming for the first trainer
                moe=True,
                batch_size=2, # Small batch size for the dummy step
                epochs=10,
            )
            # Use a minimal version of your config
            from config.task_config import MTL_TASK_CONFIG
            config = MTL_TASK_CONFIG # Use your actual config object
            config.output_folder = temp_path

            # Create and setup the first trainer
            trainer1 = Trainer(config=config, args=args1)
            trainer1.setup()

            # IMPORTANT: Run one dummy training step to initialize optimizer state
            dummy_images = torch.randn(args1.batch_size, 3, 224, 224).to(trainer1.device)
            dummy_labels = torch.randint(0, 2, (args1.batch_size, len(config.tasks))).to(trainer1.device)
            
            trainer1.optimizer.zero_grad()
            loss, _, _ = trainer1._process_batch(dummy_images, dummy_labels)
            loss.backward()
            trainer1.optimizer.step()
            trainer1.scheduler.step()

            # 2. ACT (Save the Checkpoint)
            # ------------------------------------
            print(f"\n--- Saving checkpoint to {checkpoint_path} ---")
            saving_epoch = 5
            trainer1.mtl_probe.save(
                path=checkpoint_path,
                epoch=saving_epoch,
                optimizer=trainer1.optimizer,
                scheduler=trainer1.scheduler
            )

            # 3. ARRANGE (Setup Second Trainer for Loading)
            # ------------------------------------
            print("\n--- Setting up Trainer 2 for loading ---")
            
            # New args instance pointing to the checkpoint
            args2 = SimpleNamespace(
                version='google/Siglip2-base-patch16-224',
                ckpt_path=None,
                learning_rate=0.01,
                resume_from_ckpt=str(checkpoint_path), # Point to the saved file
                moe=True,
                batch_size=2,
                epochs=10
            )

            # Create a completely new trainer instance
            trainer2 = Trainer(config=config, args=args2)
            # The .setup() method contains the loading logic
            trainer2.setup()

            # 4. ASSERT (Verify the state was restored)
            # ------------------------------------
            print("\n--- Asserting restored state ---")

            # Test 1: Start epoch is correct
            assert trainer2.start_epoch == saving_epoch, f"Epoch mismatch: Expected {saving_epoch}, got {trainer2.start_epoch}"
            print(f"[PASS] Start epoch correctly loaded as {trainer2.start_epoch}.")

            # Test 2: Model weights are identical
            for name, param1 in trainer1.mtl_probe.named_parameters():
                if param1.requires_grad:
                    print(f"Checking: {name:<50} ... ")
                param2 = trainer2.mtl_probe.state_dict()[name]
                assert torch.equal(param1, param2), f"Weight mismatch in parameter: {name}"
            print("[PASS] Model weights are identical.")
            
            # Test 3: Optimizer state is restored (we'll check a key property)
            # Note: Comparing dicts can be tricky. A good proxy is comparing a hyperparameter.
            assert trainer1.optimizer.state_dict()['param_groups'][0]['lr'] == trainer2.optimizer.state_dict()['param_groups'][0]['lr'], "Optimizer LR mismatch"
            print("[PASS] Optimizer learning rate is identical.")

            # Test 4: Scheduler state is restored
            assert trainer1.scheduler.state_dict() == trainer2.scheduler.state_dict(), "Scheduler state mismatch"
            print("[PASS] Scheduler state is identical.")

            print("\n✅ All tests passed!")

    test_trainer_save_and_load()