""" This files contain the config logic for the probing and multi-task learning tasks
The variable TASK_REGISTRY and MTL_TASK_CONFIG passed to trainers and should be modified as required"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import  Dict, List, Type
import numpy as np
import torch
import torch.nn as nn
from utils.datasets import AgeDataset, EmotionDataset, GenderDataset, BaseDataset

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

    # workers for dataloader
    num_workers: int = 16
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
    
    # workers for dataloader
    num_workers: int = 8
    ignore_index: int = -100 # default value for masked loss in nn.CrossEntropyLoss
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




# --------------------- CONFIGS VARIABLES --------------------- #
MTL_TASK_CONFIG = MTLConfig(
    tasks=[
        Task(name='Age', class_labels=["0-2","3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"], criterion=nn.CrossEntropyLoss, weight=1.0, use_weighted_loss=True),
        Task(name='Gender', class_labels=["Male", "Female"], criterion=nn.CrossEntropyLoss, weight=1.0),
        Task(name='Emotion', class_labels=["Surprise", "Fear", "Disgust", "Happy", "Sad", "Angry", "Neutral"], criterion=nn.CrossEntropyLoss, weight=1.0, use_weighted_loss=True)
    ],
    output_folder=Path('./experiment'),
    dataset_root=Path("/user/asessa/dataset tesi/"), 
    train_csv=Path("/user/asessa/test_folder/train/train.csv"),
    val_csv=Path("/user/asessa/test_folder/val/validation.csv"),
    test_csv=Path("/user/asessa/dataset tesi/mtl_test.csv"),
    use_uncertainty_weighting=True,
    use_grad_norm=False
)

"""
train_csv=Path("/user/asessa/test_folder/train/train.csv"),
val_csv=Path("/user/asessa/test_folder/val/validation.csv"),
test_csv=Path("/user/asessa/dataset tesi/mtl_test.csv"),

train_csv=/user/asessa/dataset tesi/small_train.csv
test_csv=Path("/user/asessa/dataset_labels/test/emotion/raf-db.csv"),

"""

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
            target_samples_per_class_train=25000,
            target_samples_per_class_val=5000
        ),
        dataset_root=Path("/user/asessa/dataset tesi/"),
        csv_path=Path("/user/asessa/dataset tesi/gender_labels_cropped.csv"),
        test_csv_path=Path('/user/asessa/dataset_labels/test/age/utk_fairface.csv'),
        output_folder=Path('./experiments/gender_classification'),
    ),
}

