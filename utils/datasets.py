"""
datasets.py contains the logic to handle the different datasets.

It expect the following setup:
The following CSV files contain respectively each samples with age, emotion and gender labels from the train folders:
'age_labels.csv',
'emotion_labels.csv',
'gender_labels.csv'



They all have the same header:
Path,Gender,Age,Ethnicity,Facial Emotion,Identity
If a sample is missing a label the value will be -100

The mapping for the values is the following:
age_id2label = ['0-2','3-9','10-19','20-29','30-39','40-49','50-59','60-69','more than 70'] (so 0-2 is mapped to 0, '3-9' is mapped to 1 ...)
gender_id2label = ['Male','Female']
emotion_id2label = ["Surprise", "Fear", "Disgust", "Happy", "Sad", "Angry", "Neutral"]


For testing use the following CSV:
Age     -> labels_test_utk.csv,  label_test_vgg.csv
Gender  -> labels_test_utk.csv,  label_test_vgg.csv
Emotion -> labels_test_raf.csv

"""

from abc import ABC, abstractmethod
import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.data import random_split
from torch.utils.data import Subset
from torch.utils.data import WeightedRandomSampler

PATH_COLUMN = 0
GENDER_COLUMN = 1
AGE_COLUMN = 2 
EMOTION_COLUMN = 4
age_id2label = ['0-2','3-9','10-19','20-29','30-39','40-49','50-59','60-69','70+'] 
gender_id2label = ['Male','Female']
emotion_id2label = ["Surprise", "Fear", "Disgust", "Happy", "Sad", "Angry", "Neutral"]

class BaseDataset(Dataset, ABC):
    """Abstract base class for image datasets with CSV labels."""
    
    def __init__(self, 
                 csv_path,
                 transform,
                 root_dir='/user/asessa/dataset tesi/', 
                 target_per_label=None,
                 stratify_on = 'Age'):
        
        self.labels_df = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.transform = transform
        
        # Keep a fraction of the dataset to test quickly
        if target_per_label is not None:
            print(f"Original value counts:\n{self.labels_df[stratify_on].value_counts().sort_index()}\n")
            
            self.labels_df = self.dynamic_stratified_sample(
                df=self.labels_df,
                strata_col=stratify_on,
                target_samples=target_per_label
            )
            
            print(f"Value counts after dynamic sampling (target={target_per_label}):\n{self.labels_df[stratify_on].value_counts().sort_index()}")
    

    def dynamic_stratified_sample(self, df, strata_col, target_samples):
        """
        Samples a DataFrame to a target number of samples per class in a given column.
        - Classes with > target_samples are undersampled.
        - Classes with <= target_samples are kept completely (oversampled relative to others).
        """
        
        def sample_group(group):
            if len(group) > target_samples:
                return group.sample(n=target_samples, random_state=42)
            else:
                return group

        return df.groupby(strata_col, group_keys=False).apply(sample_group)

    def __len__(self):
        return len(self.labels_df)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Load image
        relative_img_path = self.labels_df.iloc[idx, PATH_COLUMN]
        img_path = self.root_dir + relative_img_path
        image = Image.open(img_path)
        
        # Get labels using abstract method
        labels = self.get_labels(idx)
        
        return self.transform(image), labels
    
    @abstractmethod
    def get_labels(self, idx):
        """Abstract method to extract labels from the dataframe."""
        pass

    @abstractmethod
    def get_inverse_weight(self):
        """Abstract method to obtain weight for Weighted Cross Entropy."""
        pass

class AgeDataset(BaseDataset):
    """Dataset for age prediction tasks."""
    
    def get_labels(self, idx):
        return torch.tensor(self.labels_df.iloc[idx, AGE_COLUMN], 
                          dtype=torch.long)

    def get_inverse_weight(self):
        """method to obtain weight for Weighted Cross Entropy."""
        class_counts = self.labels_df.iloc[:, AGE_COLUMN].value_counts().sort_index()
        total_samples = len(self.labels_df)
        weights = total_samples / (len(age_id2label) * class_counts)
        return torch.tensor(weights.values, dtype=torch.float)

class GenderDataset(BaseDataset):
    """Dataset for gender classification tasks."""
    
    def get_labels(self, idx):
        return torch.tensor(self.labels_df.iloc[idx, GENDER_COLUMN], 
                          dtype=torch.long)

    def get_inverse_weight(self):
        """method to obtain weight for Weighted Cross Entropy."""
        class_counts = self.labels_df.iloc[:, GENDER_COLUMN].value_counts().sort_index()
        total_samples = len(self.labels_df)
        weights = total_samples / (len(gender_id2label) * class_counts)
        return torch.tensor(weights.values, dtype=torch.float)

class EmotionDataset(BaseDataset):
    """Dataset for emotion classification tasks."""
    
    def get_labels(self, idx):
        return torch.tensor(self.labels_df.iloc[idx, EMOTION_COLUMN], 
                          dtype=torch.long)
    
    def get_inverse_weight(self):
        """method to obtain weight for Weighted Cross Entropy."""
        class_counts = self.labels_df.iloc[:, EMOTION_COLUMN].value_counts().sort_index()
        total_samples = len(self.labels_df)
        weights = total_samples / (len(emotion_id2label) * class_counts)
        return torch.tensor(weights.values, dtype=torch.float)

class CombinedDataset(Dataset):
    """Dataset for multitask learning

    Use the following sampler when instanciating to achieve balanced sampling:
    sampler_weights = dataset.get_sampler_weights()
    sampler = WeightedRandomSampler(
        weights=sampler_weights,
        num_samples=len(sampler_weights),
        replacement=True
    )
    
    """
    def __init__(self,
                 age_csv_path,
                 gender_csv_path,
                 emotion_csv_path,
                 transform,
                 root_dir='/user/asessa/dataset tesi/',
                 keep=0.2):

        # age_df = pd.read_csv(age_csv_path)
        # the gender csv contains every sample of the age csv
        gender_df = pd.read_csv(gender_csv_path)
        if keep <= 1.0:
            gender_df = gender_df.sample(frac=keep)

        emotion_df = pd.read_csv(emotion_csv_path)

        # to be able to oversample 
        gender_df['source'] = 'gender'
        emotion_df['source'] = 'emotion'

        # there should be no duplicates
        self.labels_df = pd.concat([emotion_df, gender_df], ignore_index=True).drop_duplicates(subset=['Path'])

        
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # Load image
        # Column 0 in the merged dataframe is 'Path'
        relative_img_path = self.labels_df.iloc[idx, 0]
        img_path = self.root_dir + relative_img_path
        image = Image.open(img_path)

        age_label = self.labels_df.iloc[idx, AGE_COLUMN]
        gender_label = self.labels_df.iloc[idx, GENDER_COLUMN]
        emotion_label = self.labels_df.iloc[idx, EMOTION_COLUMN]
        
        labels = torch.tensor([gender_label, age_label, emotion_label], dtype=torch.long)
        
        return self.transform(image), labels
    
    def get_sampler_weights(self, desired_emotion_ratio=1/3):
        """
        Calculates weights for each sample to be used with WeightedRandomSampler.
        Samples from the emotion dataset will have a higher weight.
        """
        source_counts = self.labels_df['source'].value_counts()
        num_emotion = source_counts.get('emotion', 0)
        num_other = source_counts.get('gender', 0)


        # Using the derived formula: w_e = N_o / (target_ratio_demonimator * N_e)
        # For a ratio of 1/3, the formula is w_e = N_o / (2 * N_e)
        # More generally: desired_ratio = (N_e * w_e) / (N_e * w_e + N_o)
        # w_e = (desired_emotion_ratio * num_other) / (num_emotion * (1 - desired_emotion_ratio))
        emotion_weight = (desired_emotion_ratio * num_other) / (num_emotion * (1 - desired_emotion_ratio))

        # Assign weight 1.0 to other samples and the calculated weight to emotion samples
        source = self.labels_df['source']
        weights = [emotion_weight if s == 'emotion' else 1.0 for s in source]

        return torch.DoubleTensor(weights)
    
    def get_inverse_weights(self):
        """
        Method to obtain weights for Weighted Cross Entropy for each task.
        This is useful for handling class imbalance in a multi-task setting.
        
        Returns:
            dict: A dictionary containing weight tensors for 'Age', 'Gender', and 'Emotion'.
        """
        weights = {}
        total_samples = len(self.labels_df)

        # Age weights
        class_counts_age = self.labels_df.iloc[:, AGE_COLUMN].value_counts().sort_index()
        weights_age = total_samples / (len(age_id2label) * class_counts_age)
        weights['Age'] = torch.tensor(weights_age.values, dtype=torch.float)

        # Gender weights
        class_counts_gender = self.labels_df.iloc[:, GENDER_COLUMN].value_counts().sort_index()
        weights_gender = total_samples / (len(gender_id2label) * class_counts_gender)
        weights['Gender'] = torch.tensor(weights_gender.values, dtype=torch.float)

        # Emotion weights
        class_counts_emotion = self.labels_df.iloc[:, EMOTION_COLUMN].value_counts().sort_index()
        weights_emotion = total_samples / (len(emotion_id2label) * class_counts_emotion)
        weights['Emotion'] = torch.tensor(weights_emotion.values, dtype=torch.float)
        
        return weights




def get_loaders(full_dataset, generator, batch_size, split = [0.8,0.2]):
    """Return train and validation loader given a dataset (do not use for Combined)"""
    size = len(full_dataset)

    train_size = int(size * split[0])
    val_size = size - train_size

    print(f'Whole dataset = {size}\nTrain len = {train_size}\nVal len = {val_size}')
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=generator
    )
    print('puzzo')
    train_loader = DataLoader(train_dataset,  batch_size=batch_size,shuffle=True,
                                 num_workers=4, pin_memory=False)
    val_loader = DataLoader(val_dataset,  batch_size=batch_size,shuffle=True,
                                 num_workers=4, pin_memory=False)


    return train_loader, val_loader

if __name__ == '__main__':
    my_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    age_csv_path ='/user/asessa/dataset tesi/datasets_with_standard_labels/age_labels.csv'
    emotion_csv_path ='/user/asessa/dataset tesi/datasets_with_standard_labels/emotion_labels.csv'
    gender_csv_path ='/user/asessa/dataset tesi/datasets_with_standard_labels/gender_labels.csv'
    
    dataset = CombinedDataset(age_csv_path, gender_csv_path, emotion_csv_path, my_transforms)
    sampler_weights = dataset.get_sampler_weights()
    sampler = WeightedRandomSampler(
        weights=sampler_weights,
        num_samples=len(sampler_weights),
        replacement=True
    )


    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        sampler=sampler
    )

    # Start timing

    i = 0
    for images, labels in dataloader:
        print(labels)

        i += 1
        if i == 10:
            break

    