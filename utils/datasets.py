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
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.data import random_split
from torch.utils.data import Subset
from torch.utils.data import WeightedRandomSampler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
MISSING_LABEL = -100
PATH_COLUMN = 0
GENDER_COLUMN = 1
AGE_COLUMN = 2 
EMOTION_COLUMN = 4
age_id2label = ['0-2','3-9','10-19','20-29','30-39','40-49','50-59','60-69','70+'] 
gender_id2label = ['Male','Female']
emotion_id2label = ["Surprise", "Fear", "Disgust", "Happy", "Sad", "Angry", "Neutral"]

def get_split(path_df, stratify_column='Age', test_split=0.2):
    full_df = pd.read_csv(path_df)

    train_df, val_df = train_test_split(
        full_df, 
        test_size=test_split, 
        random_state=42, 
        stratify=full_df[stratify_column]
    )
    print(f'[DATASET] Train df = {len(train_df)}; Validation df = {len(val_df)}')
    return train_df, val_df


class BaseDataset(Dataset, ABC):
    """Abstract base class for image datasets with CSV labels."""
    
    def __init__(self, 
                 df,
                 transform,
                 root_dir='/user/asessa/dataset tesi/', 
                 stratify_on = 'Age',
                 return_path=False):
        
        self.data_pool_df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform
        self.active_df = self.data_pool_df.copy()
        self.stratify_on = stratify_on
        self.return_path = return_path
    
    """
    def resample(self, target_samples_per_class):
        print(f"\n[Dataset] Re-sampling data with target of {target_samples_per_class} per class...")
        
        def sample_group(group):
            if len(group) > target_samples_per_class:
                return group.sample(n=target_samples_per_class, random_state=np.random.randint(0,1000))
            else:
                return group

        self.labels_df = self.full_labels_df.groupby(
            self.stratify_on, group_keys=False
        ).apply(sample_group)
        
        print(f"[Dataset] New dataset size: {len(self.labels_df)}")
    """

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

    def resample(self, target_samples_per_class: int):
        """
        replaces the active DataFrame by resampling from the whole datapool
        """
        if target_samples_per_class is None:
            self.active_df = self.data_pool_df.copy()
            return
            
        grouped = self.data_pool_df.groupby(self.stratify_on)
        resampled_indices = []

        for name, group in grouped:
            if len(group) > target_samples_per_class:
                # Sample indices, which is faster than sampling rows
                resampled_indices.extend(group.sample(n=target_samples_per_class, random_state=np.random.randint(0,10000)).index)
            else:
                resampled_indices.extend(group.index)

        # Use .loc for fast slicing based on the collected indices
        self.active_df = self.data_pool_df.loc[resampled_indices].reset_index(drop=True)
        
        # print(f"[Resampled] New counts:\n{self.active_df[self.stratify_on].value_counts().sort_index()}")

    def __len__(self):
        return len(self.active_df)
    
    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
            # Load image
        relative_img_path = self.active_df.iloc[idx, PATH_COLUMN]
        img_path = self.root_dir / relative_img_path
        image = Image.open(img_path)
            
        # Get labels using abstract method
            
        labels = self.get_labels(idx)
        if self.return_path:
            return self.transform(image), labels, str(img_path)

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
        return torch.tensor(self.active_df.iloc[idx, AGE_COLUMN], 
                          dtype=torch.long)

    def get_inverse_weight(self):
        """method to obtain weight for Weighted Cross Entropy."""
        class_counts = self.active_df.iloc[:, AGE_COLUMN].value_counts().sort_index()
        total_samples = len(self.active_df)
        weights = total_samples / (len(age_id2label) * class_counts)
        return torch.tensor(weights.values, dtype=torch.float)

class GenderDataset(BaseDataset):
    """Dataset for gender classification tasks."""
    
    def get_labels(self, idx):
        return torch.tensor(self.active_df.iloc[idx, GENDER_COLUMN], 
                          dtype=torch.long)

    def get_inverse_weight(self):
        """method to obtain weight for Weighted Cross Entropy."""
        class_counts = self.active_df.iloc[:, GENDER_COLUMN].value_counts().sort_index()
        total_samples = len(self.active_df)
        weights = total_samples / (len(gender_id2label) * class_counts)
        return torch.tensor(weights.values, dtype=torch.float)

class EmotionDataset(BaseDataset):
    """Dataset for emotion classification tasks."""
    
    def get_labels(self, idx):
        return torch.tensor(self.active_df.iloc[idx, EMOTION_COLUMN], 
                          dtype=torch.long)
    
    def get_inverse_weight(self):
        """method to obtain weight for Weighted Cross Entropy."""
        class_counts = self.active_df.iloc[:, EMOTION_COLUMN].value_counts().sort_index()
        total_samples = len(self.active_df)
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
                 gender_csv_path,
                 emotion_csv_path,
                 transform,
                 root_dir='/user/asessa/dataset tesi/',
                 keep=0.1,
                 return_path=False):

        # age_df = pd.read_csv(age_csv_path)
        # the gender csv contains every sample of the age csv
        # so gender df is used also for age
        self.original_gender_df = pd.read_csv(gender_csv_path)
        self.original_emotion_df = pd.read_csv(emotion_csv_path)
        self.keep = keep
        self.root_dir = root_dir
        self.transform = transform
        
        # Initial data setup
        self.new_epoch_resample()

    def _load_and_resample_gender_df(self):
        """Internal method to load and resample the gender dataframe."""
        if self.keep <= 1.0:
            return self.original_gender_df.sample(frac=self.keep)
        return self.original_gender_df.copy()

    def new_epoch_resample(self):
        """
        Resamples the gender dataframe and rebuilds the combined labels dataframe.
        Call this method at the start of each new epoch.
        """
        gender_df = self._load_and_resample_gender_df()
        gender_df['source'] = 'gender'
        
        emotion_df = self.original_emotion_df.copy()
        emotion_df['source'] = 'emotion'
        # there should be no duplicates
        self.labels_df = pd.concat([emotion_df, gender_df], ignore_index=True).drop_duplicates(subset=['Path'])

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
        
        labels = torch.tensor([age_label, gender_label, emotion_label], dtype=torch.long)
        
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
    
    def get_inverse_weights_loss(self):
        """
        Calculates inverse weights for the current state of the dataset,
        """
        weights = {}

        # --- Age weights ---
        valid_age_labels = self.labels_df[self.labels_df.iloc[:, AGE_COLUMN] != MISSING_LABEL]
        total_valid_age = len(valid_age_labels)
        if total_valid_age > 0:
            class_counts_age = valid_age_labels.iloc[:, AGE_COLUMN].value_counts()
            weights_age = torch.zeros(len(age_id2label))
            for class_idx, count in class_counts_age.items():
                weights_age[int(class_idx)] = total_valid_age / (len(age_id2label) * count)
            weights['Age'] = weights_age

        # --- Gender weights ---
        valid_gender_labels = self.labels_df[self.labels_df.iloc[:, GENDER_COLUMN] != MISSING_LABEL]
        total_valid_gender = len(valid_gender_labels)
        if total_valid_gender > 0:
            class_counts_gender = valid_gender_labels.iloc[:, GENDER_COLUMN].value_counts()
            weights_gender = torch.zeros(len(gender_id2label))
            for class_idx, count in class_counts_gender.items():
                weights_gender[int(class_idx)] = total_valid_gender / (len(gender_id2label) * count)
            weights['Gender'] = weights_gender

        # --- Emotion weights ---
        valid_emotion_labels = self.labels_df[self.labels_df.iloc[:, EMOTION_COLUMN] != MISSING_LABEL]
        total_valid_emotion = len(valid_emotion_labels)
        if total_valid_emotion > 0:
            class_counts_emotion = valid_emotion_labels.iloc[:, EMOTION_COLUMN].value_counts()
            weights_emotion = torch.zeros(len(emotion_id2label))
            for class_idx, count in class_counts_emotion.items():
                weights_emotion[int(class_idx)] = total_valid_emotion / (len(emotion_id2label) * count)
            weights['Emotion'] = weights_emotion
        
        return weights
    
    def get_inverse_weights_loss_mc(self, num_simulations=50):
        """
        Calculates robust inverse weights using a Monte Carlo simulation.
        Returns a dictionary containing averaged weight tensors for 'Age', 'Gender', and 'Emotion'.
        """
        print(f"Running Monte Carlo simulation for inverse weights with {num_simulations} iterations...")
        
        sum_weights = {
            'Age': torch.zeros(len(age_id2label)),
            'Gender': torch.zeros(len(gender_id2label)),
            'Emotion': torch.zeros(len(emotion_id2label))
        }
        sim_counts = {'Age': 0, 'Gender': 0, 'Emotion': 0}

        for _ in tqdm(range(num_simulations), desc="MC Simulation"):
            self.new_epoch_resample()
            current_weights = self.get_inverse_weights_loss()
            
            # Add to the running total for each task if weights were calculated
            for task, weights_tensor in current_weights.items():
                sum_weights[task] += weights_tensor
                sim_counts[task] += 1

        # Average the weights, only dividing by the number of simulations where weights were present
        avg_weights = {}
        for task in sum_weights.keys():
            avg_weights[task] = sum_weights[task] / sim_counts[task]


        self.new_epoch_resample()
        print("Monte Carlo simulation complete.")
        return avg_weights


def resample(dataset_train, dataset_val, batch_size, target_samples_per_class_train, target_samples_per_class_val, num_workers):
    """To call at the start of each new epoch, resamples the whole dataset and returns new dataloaders"""
    dataset_train.resample(target_samples_per_class=target_samples_per_class_train)

    train_loader = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    dataset_val.resample(target_samples_per_class=target_samples_per_class_val)

    val_loader = DataLoader(
        dataset_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader

def get_loaders(full_dataset, generator, batch_size, split = [0.8,0.2]):
    """Return train and validation loader given a dataset (do not use for Combined)"""
    size = len(full_dataset)

    train_size = int(size * split[0])
    val_size = size - train_size

    print(f'Whole dataset = {size}\nTrain len = {train_size}\nVal len = {val_size}')
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=generator
    )
    train_loader = DataLoader(train_dataset,  batch_size=batch_size,shuffle=True,
                                 num_workers=16, pin_memory=True)
    val_loader = DataLoader(val_dataset,  batch_size=batch_size,shuffle=True,
                                 num_workers=16, pin_memory=True)


    return train_loader, val_loader

import torchvision.transforms as transforms
import math

FIX = 0

PATH_COLUMN = 1 + FIX
GENDER_COLUMN = 2 + FIX
AGE_COLUMN = 3 + FIX 
EMOTION_COLUMN = 5 + FIX

class MTLDataset(Dataset):
    def __init__(self, csv_path, transform , balance_on="Facial Emotion", augment=True, root_dir= "/user/asessa/dataset tesi/", balance=True, return_path=False, csv_has_index=False):
        self.labels_df = pd.read_csv(csv_path)
        self.transform = transform 
        self.root_dir = root_dir
        if augment: 
            self.transform = transforms.Compose([ 
                transforms.RandomHorizontalFlip(), 
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomRotation(degrees=10),
                transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
                *transform.transforms,
                transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3)) 
            ])
        if balance: 
            self._balance_dataset(column_name=balance_on)
        self.return_path = return_path
        
        self.path_column = 1 if csv_has_index else 0
        self.gender_column = 2 if csv_has_index else 1
        self.age_column = 3 if csv_has_index else 2
        self.emotion_column = 5 if csv_has_index else 4


    def _balance_dataset(self, column_name, target_percentage=0.33):
        """Duplicates samples with valid labels in the specified column to meet the target percentage."""
        print(f"[DATASET] Balancing dataset on column: '{column_name}' to reach at least {target_percentage*100:.1f}% valid samples.")
            
        # Identify valid and invalid samples
        valid_samples_df = self.labels_df[self.labels_df[column_name] != MISSING_LABEL ]
            
        num_valid = len(valid_samples_df)
        num_total = len(self.labels_df)
        current_percentage = num_valid / num_total 

        print(f"[DATASET] Initial state: {num_valid} valid samples out of {num_total} ({current_percentage*100:.1f}%).")

        # Check if balancing is needed
        if current_percentage >= target_percentage:
            print("[DATASET] Target percentage already met. No balancing needed.")
            return

        # Calculate how many samples need to be added
        # Formula derived from: (num_valid + N) / (num_total + N) = target
        num_to_add = math.ceil((target_percentage * num_total - num_valid) / (1 - target_percentage))
            
        if num_to_add <= 0:
            return

        print(f"[DATASET] Need to add {num_to_add} samples with valid '{column_name}' labels.")

        # Duplicate random samples from the valid ones
 
        samples_to_add = valid_samples_df.sample(n=num_to_add, replace=True)
                
        # Append the new samples and shuffle the dataset
        self.labels_df = pd.concat([self.labels_df, samples_to_add], ignore_index=True)
        self.labels_df = self.labels_df.sample(frac=1).reset_index(drop=True) # Shuffle
                
        new_total = len(self.labels_df)
        new_valid = len(self.labels_df[self.labels_df[column_name] != MISSING_LABEL ])
        print(f"[DATASET] Balancing complete. New total samples: {new_total}. New valid samples: {new_valid} ({(new_valid/new_total)*100:.1f}%).")

    def get_inverse_weights_loss(self):

        weights = {}
        # --- Age weights ---
        valid_age_labels = self.labels_df[self.labels_df.iloc[:, self.age_column] != MISSING_LABEL]
        total_valid_age = len(valid_age_labels)
        if total_valid_age > 0:
            class_counts_age = valid_age_labels.iloc[:, self.age_column].value_counts()
            weights_age = torch.zeros(len(age_id2label))
            for class_idx, count in class_counts_age.items():
                weights_age[int(class_idx)] = total_valid_age / (len(age_id2label) * count)
            weights['Age'] = weights_age

        # --- Gender weights ---
        valid_gender_labels = self.labels_df[self.labels_df.iloc[:, self.gender_column] != MISSING_LABEL]
        total_valid_gender = len(valid_gender_labels)
        if total_valid_gender > 0:
            class_counts_gender = valid_gender_labels.iloc[:, self.gender_column].value_counts()
            weights_gender = torch.zeros(len(gender_id2label))
            for class_idx, count in class_counts_gender.items():
                weights_gender[int(class_idx)] = total_valid_gender / (len(gender_id2label) * count)
            weights['Gender'] = weights_gender

        # --- Emotion weights ---
        valid_emotion_labels = self.labels_df[self.labels_df.iloc[:, self.emotion_column] != MISSING_LABEL]
        total_valid_emotion = len(valid_emotion_labels)
        if total_valid_emotion > 0:
            class_counts_emotion = valid_emotion_labels.iloc[:, self.emotion_column].value_counts()
            weights_emotion = torch.zeros(len(emotion_id2label))
            for class_idx, count in class_counts_emotion.items():
                weights_emotion[int(class_idx)] = total_valid_emotion / (len(emotion_id2label) * count)
            weights['Emotion'] = weights_emotion
        
        return weights



    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # Load image
        # Column 0 in the merged dataframe is 'Path'
        relative_img_path = self.labels_df.iloc[idx, self.path_column]

        img_path = self.root_dir / relative_img_path
        image = Image.open(img_path)

        age_label = self.labels_df.iloc[idx, self.age_column]
        gender_label = self.labels_df.iloc[idx, self.gender_column]
        emotion_label = self.labels_df.iloc[idx, self.emotion_column]

  

        labels = torch.tensor([int(age_label), int(gender_label), int(emotion_label)], dtype=torch.long)
        if self.return_path:
            return self.transform(image), labels, str(img_path)
        return self.transform(image), labels

if __name__ == '__main__':

    
    my_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = MTLDataset(csv_path="/user/asessa/test_folder/train/train.csv",transform=my_transforms)
    dataloader = DataLoader(dataset=dataset, batch_size=64)
    loader = tqdm(dataloader)
    total_samples_processed = 0
    valid_emotion_samples = 0
    for im, labels in loader:
        emotion_labels = labels[:, 2] # Emotion is the 3rd element (index 2)
            
        valid_count_in_batch = (emotion_labels != -100).sum().item()
            
        valid_emotion_samples += valid_count_in_batch
        total_samples_processed += len(labels)

    final_percentage = valid_emotion_samples / total_samples_processed 
    print("\n--- VERIFICATION RESULTS ---")
    print(f"Total samples in final dataset: {total_samples_processed}")
    print(f"Samples with valid emotion label: {valid_emotion_samples}")
    print(f"Final percentage of valid samples: {final_percentage:.2%}")

    exit()
    age_csv_path ='/user/asessa/dataset tesi/datasets_with_standard_labels/age_labels_cropped.csv'
    emotion_csv_path ='/user/asessa/dataset tesi/emotion_labels_cropped.csv'
    gender_csv_path ='/user/asessa/dataset tesi/gender_labels_cropped.csv'
    
    dataset = CombinedDataset(gender_csv_path, emotion_csv_path, my_transforms, keep=1.0)

    # print(dataset.get_inverse_weights_loss_mc())

    sampler_weights = dataset.get_sampler_weights()
    print(f'{len(sampler_weights)}\n{sampler_weights.shape}')
    sampler = WeightedRandomSampler(
        weights=sampler_weights,
        num_samples=len(sampler_weights),
        replacement=True
    )


    dataloader = DataLoader(
        dataset,
        batch_size=128,
    )

    # Start timing

    loader = tqdm(dataloader)
    for images, labels in loader:
        pass
    