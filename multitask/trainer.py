import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any,  Dict, List, Tuple, Type

import matplotlib.pyplot as plt
import torch
from dotenv import load_dotenv
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import torch.optim as optim
from torch.optim.lr_scheduler import  CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader,  WeightedRandomSampler
from tqdm import tqdm
import numpy as np

# Environment and Path Setup
load_dotenv()
REPO_PATH = os.getenv("REPO_PATH")
if REPO_PATH:
    sys.path.append(REPO_PATH)

from probing.probe import Probe
from utils.commons import log_to_disk, get_backbone
from utils.datasets import get_split, resample, MTLDataset
from config.task_config import TaskConfig, MTLConfig
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
        self.version_name = 'Siglip2' if 'google' in args.version else args.version
        self.saving_interval = 10
        self.start_epoch = 0

        # These will be initialized in self.setup()
        self.mtl_probe = None
        self.transform = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.optimizer = None
        self.scheduler = None
        self.criterions = {}
        
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
            use_moe=self.args.moe
        ).to(self.device)
        
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.mtl_probe.parameters()),
            lr=self.args.learning_rate
        )
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2)
        
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

        self.train_loader = DataLoader(dataset_train, batch_size=self.args.batch_size, num_workers=self.config.num_workers, pin_memory=True)
        if self.config.num_workers == 1:
            print(f'!!!!!!NUM WORKERS IS ONLY 1!!!!!!!')
            # exit()
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
            balance=False
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
        
        

    """ 
    def _setup_loaders_old(self):
        if self.dataset_train == None:
            self.dataset_train = self.config.dataset_class(
                root_dir=self.args.dataset_root, 
                gender_csv_path=self.args.csv_path_gender, 
                emotion_csv_path=self.args.csv_path_emotions, 
                transform=self.transform,
                keep=0.1
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
    """

    def _get_save_path(self, epoch: int, is_head_only: bool = False) -> Path:
        """Generates a consistent path for saving checkpoints."""
        task_name = 'mtl'
        suffix = f"head+{str(epoch)}" if is_head_only else str(epoch)
        filename = f"{task_name}_{self.version_name}_{suffix}.pt"
        return self.config.output_folder / 'ckpt' / filename
    

    def train(self):
        """Runs the main training loop."""
        self.setup()
        print("\n--- Starting Training ---")
        self.config.output_folder.mkdir(parents=True, exist_ok=True)
        (self.config.output_folder / 'ckpt').mkdir(exist_ok=True)
        
        minimum_val_accuracy = float('inf')

        for epoch in range(self.start_epoch, self.args.epochs):
            print(f"\nEpoch {epoch + 1}/{self.args.epochs}")
            
            train_results = self._run_epoch(self.train_loader, is_training=True, epoch_idx=epoch)
            print(f"TRAINING Avg Loss: {train_results['avg_loss']:.4f}, Task Losses: {train_results['task_losses']}")
            
            val_results = self._run_epoch(self.val_loader, is_training=False, description="Validating")
            print(f"VALIDATION Avg Loss: {val_results['avg_loss']:.4f}, Task Accuracies: {val_results['accuracies']}")

            for task_name, cm in val_results['confusion_matrices'].items():
                self.save_confusion_matrix(cm, task_name)
            
            self._log_epoch_results(epoch, train_results, val_results)
            
            val_accuracy = val_results['avg_accuracy']
            if (epoch + 1) % self.saving_interval == 0 or val_accuracy < minimum_val_accuracy or (epoch + 1) == self.args.epochs:
                if val_accuracy < minimum_val_accuracy:
                    print(f"[TRAINER] New best validation accuracy: {val_accuracy:.5f}. Saving model.")
                    minimum_val_accuracy = val_accuracy
                
                save_path = self.config.output_folder / 'ckpt' / f"mtl_{self.version_name}_{epoch + 1}.pt"
                self.mtl_probe.save(path=str(save_path), epoch=epoch + 1, optimizer=self.optimizer, scheduler=self.scheduler)

    def test(self):
        """Evaluates the model on the test set."""
        self.setup() # Ensure model is loaded
        self._setup_test_loader()
        
        print("\n--- Starting Testing ---")
        test_results = self._run_epoch(self.test_loader, is_training=False, description="Testing")
        
        for task_name, cm in test_results['confusion_matrices'].items():
            self.save_confusion_matrix(cm, task_name, is_final=True)
        
        print("\n--- Test Results ---")
        print(f"Average Loss: {test_results['avg_loss']:.4f}")
        print(f"Average Accuracy: {test_results['avg_accuracy']:.2f}%")
        print("\nPer-Task Losses:")
        for name, loss in test_results['task_losses'].items():
            print(f"  - {name}: {loss:.4f}")
        print("\nPer-Task Accuracies:")
        for name, acc in test_results['accuracies'].items():
            print(f"  - {name}: {acc:.2f}%")


    def _log_epoch_results(self, i, accuracies, train_loss_for_tasks, val_loss_for_tasks,train_loss_avg,val_loss_avg,accuracy_avg ):
        lr = self.scheduler.get_last_lr()[0]
        accuracies_string = ",".join(f"{x:.4f}" for x in accuracies)
        loss_string_train_tasks = ",".join(f"{x:.4f}" for x in train_loss_for_tasks)
        loss_string_val_tasks = ",".join(f"{x:.4f}" for x in val_loss_for_tasks)
        log_string=f'{i+1},{train_loss_avg:.4f},{loss_string_train_tasks},{val_loss_avg:.4f},{loss_string_val_tasks},{accuracy_avg:.3f},{accuracies_string},{lr:.6f}'
        log_to_disk(self.config.output_folder, log_string,
                         f'{self.probing_type}_{self.version_name}',header=self.config.header)
        


    def _run_epoch(self, loader: DataLoader, is_training: bool, description: str = "Training", epoch_idx: int = 0) -> Dict[str, Any]:
        """
        Runs a single epoch of training or evaluation.
        
        Args:
            loader: The DataLoader for the current dataset.
            is_training: If True, performs training (backpropagation), otherwise evaluation.
            description: A string for the progress bar.
            epoch_idx: The current epoch index, used for the learning rate scheduler.
            
        Returns:
            A dictionary containing metrics for the epoch.
        """
        self.mtl_probe.train(is_training)
        
        total_loss = 0.0
        task_loss_totals = {task.name: 0.0 for task in self.config.tasks}
        task_sample_counts = {task.name: 0 for task in self.config.tasks}
        
        # Placeholders for predictions and true labels, used only in evaluation
        all_preds = {task.name: [] for task in self.config.tasks}
        all_true = {task.name: [] for task in self.config.tasks}

        pbar = tqdm(enumerate(loader), total=len(loader), desc=description)

        context = torch.no_grad() if not is_training else torch.enable_grad()
        with context:
            for i, (images, labels) in pbar:
                loss, task_losses, task_outputs = self._process_batch(images, labels)

                if is_training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    if self.scheduler:
                        self.scheduler.step(epoch_idx + i / len(loader))

                total_loss += loss.item()
                
                # Update progress bar
                task_loss_str = ", ".join([f"{name}: {l:.4f}" for name, l in task_losses.items()])
                pbar.set_description(f"{description} - Avg Loss: {total_loss / (i + 1):.4f} | Batch Losses: [{task_loss_str}]")
                if is_training and self.scheduler:
                    pbar.set_postfix(lr=f"{self.scheduler.get_last_lr()[0]:.6f}")

                # Accumulate losses and predictions
                for idx, task in enumerate(self.config.tasks):
                    target = labels[:, idx]
                    valid_mask = (target != self.config.ignore_index)
                    if not valid_mask.any():
                        continue

                    num_valid = valid_mask.sum().item()
                    task_sample_counts[task.name] += num_valid
                    task_loss_totals[task.name] += task_losses.get(task.name, 0.0) * num_valid
                    
                    if not is_training:
                        preds = task_outputs[idx][valid_mask].argmax(dim=1)
                        all_preds[task.name].extend(preds.cpu().numpy())
                        all_true[task.name].extend(target[valid_mask].cpu().numpy())

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
                    # 1/2σ² * loss + log(σ)
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
        log_string = ",".join(map(str, log_items))
        
        log_to_disk(
            self.config.output_folder,
            log_string,
            f'mtl_{self.version_name}',
            header=self.config.header
        )

    def save_confusion_matrix(self, cm: np.ndarray, task_name: str, is_final: bool = False):
        """Saves a confusion matrix plot to disk."""
        task = self.config.task_map[task_name]
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=task.class_labels)
        disp.plot()
        
        suffix = '_test_set' if is_final else ''
        cm_path = self.config.output_folder / f'cmkprobes_mtl_{self.version_name}_{task_name}{suffix}.jpg'
        
        plt.savefig(cm_path, bbox_inches='tight')
        plt.close() # Close the plot to free memory
        print(f"Confusion matrix for '{task_name}' saved to {cm_path}")

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