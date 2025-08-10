import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Type

import numpy as np
import torch
import torch.nn as nn
from dotenv import load_dotenv
from torch.utils.data import  Dataset

# Environment and Path Setup
load_dotenv()
REPO_PATH = os.getenv("REPO_PATH")
if REPO_PATH:
    sys.path.append(REPO_PATH)

# Assuming these are your custom modules
from utils.datasets import AgeDataset, EmotionDataset, GenderDataset, BaseDataset, CombinedDataset

""" 
class AgeOrdinalLoss():
    def __init__(self, num_classes=9, ordinal_loss='mae'):
        self.num_classes = num_classes
        self.ordinal_loss = nn.L1Loss() if ordinal_loss == 'mae' else nn.MSELoss()
        self.classes_range = torch.arange(num_classes).float()

    def __call__(self, logit, true_labels, return_predicted_label=False):
        self.classes_range = self.classes_range.to(logit.device)
        
        probabilities = torch.softmax(logit, dim=1)
        expected_value = torch.sum(probabilities * self.classes_range.unsqueeze(0), dim=1).float()
        predicted_label = torch.round(expected_value).long()

        loss = self.ordinal_loss(expected_value, true_labels.float())

        if return_predicted_label:
            return loss, predicted_label
        return loss


def get_classification_preds(outputs: torch.Tensor, labels: List[str]) -> List[str]:
    pred_indices = outputs.argmax(dim=1).cpu().numpy()
    return [labels[p] for p in pred_indices]

def get_age_preds(outputs: torch.Tensor, labels: List[str]) -> List[str]:
    classes_range = torch.arange(len(labels), device=outputs.device, dtype=torch.float32)
    probabilities = torch.softmax(outputs, dim=1)
    expected_values = torch.sum(probabilities * classes_range.unsqueeze(0), dim=1)
    pred_indices = torch.round(expected_values).long().cpu().numpy()
    # Clamp indices to be within the valid range
    pred_indices = pred_indices.clip(0, len(labels) - 1)
    return [labels[p] for p in pred_indices]

@dataclass
class TaskConfig:
    n_classes: int
    criterion: nn.Module
    dataset_class: Type[BaseDataset]
    labels: List[str]
    output_folder: Path
    test_set_path: Path
    stratify_column: str
    get_predictions: Callable[[torch.Tensor, List[str]], List[str]]
    true_label_map: Callable[[np.ndarray], List[str]]
    target_samples_per_class_train : int
    target_samples_per_class_val : int
    use_inverse_weights : bool

TASK_REGISTRY = {
    'age_classification': TaskConfig(
        n_classes=9,
        criterion=nn.CrossEntropyLoss(),
        dataset_class=AgeDataset,
        labels=["0-2","3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"],
        output_folder=Path('./age_outputs'),
        test_set_path=Path('/user/asessa/dataset tesi/datasets_with_standard_labels/UTKFace/test/labels_test_utk.csv'),
        stratify_column='Age',
        get_predictions=get_age_preds,
        true_label_map=lambda labels_flat, class_labels: [class_labels[l] for l in labels_flat],
        target_samples_per_class_train=1800,
        target_samples_per_class_val=180,
        use_inverse_weights=False
    ),
    'gender': TaskConfig(
        n_classes=2,
        criterion=nn.CrossEntropyLoss(),
        dataset_class=GenderDataset,
        labels=["Male", "Female"],
        output_folder=Path('./gender_outputs'),
        test_set_path=Path('/user/asessa/dataset tesi/datasets_with_standard_labels/UTKFace/test/labels_test_utk.csv'),
        stratify_column='Gender',
        get_predictions=get_classification_preds,
        true_label_map=lambda labels_flat, class_labels: [class_labels[int(l)] for l in labels_flat],
        target_samples_per_class_train=8192,
        target_samples_per_class_val=1280,
        use_inverse_weights=False
    ),
    'emotion': TaskConfig(
        n_classes=7,
        criterion=nn.CrossEntropyLoss(),
        dataset_class=EmotionDataset,
        labels=["Surprise", "Fear", "Disgust", "Happy", "Sad", "Angry", "Neutral"],
        output_folder=Path('./emotions_outputs'),
        test_set_path=Path('/user/asessa/dataset tesi/datasets_with_standard_labels/RAF-DB/test/labels_test_raf.csv'),
        stratify_column='Facial Emotion',
        get_predictions=get_classification_preds,
        true_label_map=lambda labels_flat, class_labels: [class_labels[l] for l in labels_flat],
        target_samples_per_class_train=99999, # whole dataset
        target_samples_per_class_val=999999,  # whole dataset
        use_inverse_weights=True
    ),
}


"""
@dataclass
class TaskSingle:
    """Encapsulates all configuration for a single task."""
    name: str
    class_labels: List[str]
    criterion: Type[nn.Module]
    dataset_class: Type[BaseDataset]
    stratify_column: str
    use_weighted_loss: bool = False
    
    # Task-specific data handling parameters
    target_samples_per_class_train: int = -1 # -1 indicates no resampling
    target_samples_per_class_val: int = -1

    @property
    def num_classes(self) -> int:
        return len(self.class_labels)

@dataclass
class SingleConfig:
    """A centralized configuration for the Single-Task training experiment."""
    task: TaskSingle
    dataset_root: Path
    csv_path: Path
    test_csv_path: Path
    output_folder: Path

    # Training settings
    num_workers: int = 8
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    @property
    def header(self) -> str:
        """generates the CSV header for logging."""
        return "epoch,train_loss,val_loss,val_accuracy,lr"


@dataclass
class Task:
    """Encapsulates all configuration for a single task."""
    name: str
    class_labels: List[str]
    criterion: Type[nn.Module]
    weight: float = 1.0
    use_weighted_loss: bool = False
    
    @property
    def num_classes(self) -> int:
        return len(self.class_labels)

@dataclass
class MTLConfig:
    """A centralized configuration for the Multi-Task Learning experiment."""
    tasks: List[Task]
    output_folder: Path
    dataset_root: Path
    train_csv: Path
    val_csv: Path
    test_csv: Path
    use_grad_norm: bool
    use_uncertainty_weighting: bool
    
    # These could also be moved to a separate TrainingConfig if needed
    num_workers: int = 8
    ignore_index: int = -100
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    task_map: Dict[str, Task] = field(init=False)

    def __post_init__(self):
        """Create a quick-access map from task names to task objects."""
        self.task_map = {task.name: task for task in self.tasks}
    
    @property
    def header(self) -> str:
        """Dynamically generates the CSV header based on the tasks."""
        task_names = [task.name for task in self.tasks]
        train_loss_headers = ','.join([f"train_{name}_loss" for name in task_names])
        val_loss_headers = ','.join([f"val_{name}_loss" for name in task_names])
        accuracy_headers = ','.join([f"accuracy_{name}" for name in task_names])
        return f"epoch,train_avg_loss,{train_loss_headers},val_avg_loss,{val_loss_headers},accuracy_val_avg,{accuracy_headers},lr"

# MTL configs
MTL_TASK_CONFIG = MTLConfig(
    tasks=[
        Task(name='Age', class_labels=["0-2","3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"], criterion=nn.CrossEntropyLoss, weight=1.0, use_weighted_loss=True),
        Task(name='Gender', class_labels=["Male", "Female"], criterion=nn.CrossEntropyLoss, weight=1.0),
        Task(name='Emotion', class_labels=["Surprise", "Fear", "Disgust", "Happy", "Sad", "Angry", "Neutral"], criterion=nn.CrossEntropyLoss, weight=1.0, use_weighted_loss=True)
    ],
    output_folder=Path('./outputs_pre_trained_heads'),
    dataset_root=Path("/user/asessa/dataset tesi/"), 
    train_csv=Path("/user/asessa/dataset tesi/small_train.csv"),
    val_csv=Path("/user/asessa/dataset tesi/mtl_test.csv"),
    test_csv=Path("/user/asessa/dataset tesi/datasets_with_standard_labels/UTKFace/test/labels_test_utk.csv"),
    # test_csv=Path("/user/asessa/dataset tesi/mtl_test.csv"),
    use_uncertainty_weighting=True,
    use_grad_norm=False
)

TASK_REGISTRY = {
    'age': SingleConfig(
        task=TaskSingle(
            name='age',
            class_labels=["0-2","3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"],
            criterion=nn.CrossEntropyLoss, 
            dataset_class=AgeDataset,
            stratify_column='Age',
            use_weighted_loss=True,
            target_samples_per_class_train=10000,
            target_samples_per_class_val=1000
        ),
        dataset_root=Path("/user/asessa/dataset tesi/"),
        csv_path=Path('/user/asessa/dataset tesi/age_labels_cropped.csv'),
        test_csv_path=Path('/user/asessa/dataset_labels/test/age/utk_fairface.csv'),
        output_folder=Path('./experiments/age_classification')
    ),
    'emotion': SingleConfig(
        task=TaskSingle(
            name='emotion',
            class_labels=["Surprise", "Fear", "Disgust", "Happy", "Sad", "Angry", "Neutral"],
            criterion=nn.CrossEntropyLoss, 
            dataset_class=EmotionDataset,
            stratify_column='Facial Emotion', 
            use_weighted_loss=True
        ),
        dataset_root=Path("/user/asessa/dataset tesi/"),
        csv_path=Path("/user/asessa/dataset tesi/emotion_labels_cropped.csv"),
        test_csv_path=Path("/user/asessa/dataset_labels/test/emotion/raf-db.csv"),
        output_folder=Path('./experiments/emotion_classification')
    ),
    'gender': SingleConfig(
        task=TaskSingle(
            name='gender',
            class_labels=['male', 'female'],
            criterion=nn.CrossEntropyLoss,
            dataset_class=GenderDataset,
            stratify_column='Gender',
            target_samples_per_class_train=50000,
            target_samples_per_class_val=50000
        ),
        dataset_root=Path("/user/asessa/dataset tesi/"),
        csv_path=Path("/user/asessa/dataset tesi/gender_labels_cropped.csv"),
        test_csv_path=Path('/user/asessa/dataset_labels/test/age/utk_fairface.csv'),
        output_folder=Path('./experiments/gender_classification'),
    ),
}

