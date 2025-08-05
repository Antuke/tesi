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
    """Processes standard classification outputs."""
    pred_indices = outputs.argmax(dim=1).cpu().numpy()
    return [labels[p] for p in pred_indices]

def get_age_preds(outputs: torch.Tensor, labels: List[str]) -> List[str]:
    """Processes ordinal regression outputs for age."""
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

@dataclass
class MTLConfig:
    task_names: List[str]
    age_labels: List[str]
    gender_labels: List[str]
    emotion_labels: List[str]
    test_set_gender_age: Path
    test_set_emotions: Path
    dataset_class: Type[CombinedDataset]
    criterions: List[Type[nn.Module]]
    output_folder: Path
    header: str
    class_labels_list: List[List[str]]
    task_to_labels: Dict[str, List[str]] = field(init=False)

    def __post_init__(self):
        self.task_to_labels = {
            "Age": self.age_labels,
            "Gender": self.gender_labels,
            "Emotion": self.emotion_labels
        }

MTL_TASK_CONFIG = MTLConfig(
    class_labels_list=[["0-2","3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"],["Male", "Female"] ,["Surprise", "Fear", "Disgust", "Happy", "Sad", "Angry", "Neutral"]],
    task_names=['Age','Gender','Emotion'],
    age_labels= ["0-2","3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"],
    emotion_labels=["Surprise", "Fear", "Disgust", "Happy", "Sad", "Angry", "Neutral"],
    gender_labels=["Male", "Female"],
    test_set_gender_age=Path('/user/asessa/dataset tesi/datasets_with_standard_labels/UTKFace/test/labels_test_utk.csv'),
    test_set_emotions=Path('/user/asessa/dataset tesi/datasets_with_standard_labels/RAF-DB/test/labels_test_raf.csv'),
    dataset_class=CombinedDataset,
    criterions=[nn.CrossEntropyLoss, nn.CrossEntropyLoss, nn.CrossEntropyLoss],
    output_folder=Path('./outputs'),
    header="epoch,train_avg_loss,train_age_loss,train_gender_loss,train_emotion_loss,val_avg_loss,val_age_loss,val_gender_loss,val_emotion_loss,accuracy_val_avg,accuracy_age,accuracy_gender,accuracy_emotion,lr"
) 