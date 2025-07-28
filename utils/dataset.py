import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.data import random_split
from torch.utils.data import Subset
age_group = {
    '0-2' : 0,
    '3-9' : 1,
    '10-19' : 2,
    '20-29' : 3,
    '30-39' : 4,
    '40-49' : 5,
    '50-59' : 6,
    '60-69' : 7,
    'more than 70': 75,
  
}

EMOTION_INVERSE_FREQ = [
    9.936464,   # 0.0
    40.991453,  # 1.0
    16.730233,  # 2.0
    2.488843,   # 3.0
    7.147541,   # 4.0
    16.887324,  # 5.0
    4.666883    # 6.0
]

AGE_INVERSE_FREQ = [
    76.483320,   # '0-2'
    20.765369,   # '3-9'
    18.216466,   # '10-19'
    3.750283,    # '20-29'
    3.626549,    # '30-39'
    6.127744,    # '40-49'
    8.358365,    # '50-59'
    21.637215,   # '60-69'
    80.434254    # 'more than 70'
]

GENDER_INVERSE_FREQ = [
    # 1.935615,   # male
    2.068814    # female
]

age_id2label = ['0-2','3-9','10-19','20-29','30-39','40-49','50-59','60-69','more than 70']
age_label2id = {label: idx for idx, label in enumerate(age_id2label)}



class AgeDataset(Dataset):
    def __init__(self, root_dir, csv_file='age_labels.csv', transform=None, classification=True):
        """
        
        Args:
            csv_file (string): Full path to the CSV file with annotations.
            root_dir (string): Absolute path to the directory containing the dataset folders
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.labels_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.classification = classification

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        relative_img_path = self.labels_df.iloc[idx, 0]
        format = '.jpg'

        full_img_path = os.path.join(self.root_dir, relative_img_path + format)
        

        image = Image.open(full_img_path)
        if self.classification:
            age_group_label = self.labels_df.iloc[idx, 2]
            label_id = age_label2id[age_group_label]
            labels = torch.tensor(label_id, dtype=torch.long)
        else:
            labels = torch.tensor([self.labels_df.iloc[idx, 2]], dtype=torch.float32)


        if self.transform:
            image = self.transform(image)

        return image, labels
    
class EmotionDataset(Dataset):
    def __init__(self, root_dir, csv_file='emotions_labels.csv', transform=None):
        """
        Args:
            csv_file (string): Full path to the CSV file with annotations.
            root_dir (string): Absolute path to the directory containing the dataset folders
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.labels_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform


    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        relative_img_path = self.labels_df.iloc[idx, 0]
        format = '.jpg'
        full_img_path = os.path.join(self.root_dir, relative_img_path + format)
        

        image = Image.open(full_img_path)


        labels = torch.tensor(int(self.labels_df.iloc[idx, 4]))


        if self.transform:
            image = self.transform(image)

        return image, labels

class GenderDataset(Dataset):
    def __init__(self, root_dir, csv_file='merged_labels.csv', transform=None):
        """
        Args:
            csv_file (string): Full path to the CSV file with annotations.
            root_dir (string): Absolute path to the directory containing the dataset folders
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.labels_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform


    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        relative_img_path = self.labels_df.iloc[idx, 0]
        format = '.jpg'
        full_img_path = os.path.join(self.root_dir, relative_img_path + format)
        

        image = Image.open(full_img_path)

        def float_convert(value):
            if pd.isna(value) or value == '':
                return float('nan')
            return float(value)

        labels = torch.tensor([
            float_convert(self.labels_df.iloc[idx, 1]),  # gender
        ], dtype=torch.float32)


        if self.transform:
            image = self.transform(image)

        return image, labels

class MergedDataset(Dataset):
    def __init__(self, root_dir, csv_file='merged_labels.csv', transform=None, classification=True):
        """
        Args:
            csv_file (string): Full path to the CSV file with annotations.
            root_dir (string): Absolute path to the directory containing the dataset folders.
            transform (callable, optional): Optional transform to be applied on a sample.
            classification (bool): If True, returns class indices. If False, returns floats.


        """
        self.labels_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.classification = classification
        self.ignore_index = -100

    def __len__(self):
        return len(self.labels_df)

    def _to_float(self, value):
        """Helper for regression: converts a value to float, handling NaNs."""
        if pd.isna(value) or value == '':
            return np.nan
        return float(value)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        relative_img_path = self.labels_df.iloc[idx, 0]
        full_img_path = os.path.join(self.root_dir, f"{relative_img_path}.jpg")

        image = Image.open(full_img_path).convert('RGB')

        if self.classification:

            raw_gender = self.labels_df.iloc[idx, 1]
            gender_id = self.ignore_index if pd.isna(raw_gender) else int(raw_gender)
            
            raw_age_group = self.labels_df.iloc[idx, 5]
            age_id = age_label2id.get(raw_age_group, self.ignore_index)

            raw_emotion = self.labels_df.iloc[idx, 3]
            emotion_id = self.ignore_index if pd.isna(raw_emotion) else int(raw_emotion)
            
            labels = torch.tensor([gender_id, age_id, emotion_id], dtype=torch.long)

        else:
            gender_val = self._to_float(self.labels_df.iloc[idx, 1])
            age_val = self._to_float(self.labels_df.iloc[idx, 2])
            emotion_val = self._to_float(self.labels_df.iloc[idx, 4])
            
            labels = torch.tensor([gender_val, age_val, emotion_val], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, labels


def get_loaders_broken_(full_dataset, generator, batch_size=32):
    """
    Given a dataset, returns three dataloaders, training 80%, test and validation 10%-
    It uses generator for riproducibilità
    """
    dataset_size = len(full_dataset)
    train_size = int(dataset_size * 0.8)
    val_size = int(dataset_size * 0.1)
    test_size = dataset_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader
    

def label_distribution(dataloader: DataLoader, num_batches: int, labels_names):

    label_count = {'gender': 0, 'age': 0, 'emotion': 0}
    total_samples = 0

    for batch_idx, (_, labels) in enumerate(dataloader):
        if batch_idx >= num_batches:
            break
            
        batch_size = labels.shape[0]
        total_samples += batch_size
        

        for i, label_name in enumerate(labels_names):
            non_nan_count = torch.sum(~torch.isnan(labels[:, i].float())).item()
            label_count[label_name] += non_nan_count
    
    label_probabilities = {}
    for label_name, count in label_count.items():
        label_probabilities[label_name] = count / total_samples
    

    return label_probabilities, total_samples

def get_loaders(full_dataset, generator, batch_size=32):
    """
    Given a dataset, returns three dataloaders, training 80%, test and validation 10%.
    It uses a generator for reproducibility and ensures no data leakage with duplicate samples.
    """
    unique_images = full_dataset.labels_df.iloc[:, 0].unique()
    unique_indices = np.arange(len(unique_images))
    generator_instance = np.random.default_rng(generator.initial_seed())
    generator_instance.shuffle(unique_indices)

    train_size = int(len(unique_indices) * 0.8)
    val_size = int(len(unique_indices) * 0.1)

    train_unique_indices = unique_indices[:train_size]
    val_unique_indices = unique_indices[train_size:train_size + val_size]
    test_unique_indices = unique_indices[train_size + val_size:]

    train_imgs = unique_images[train_unique_indices]
    val_imgs = unique_images[val_unique_indices]
    test_imgs = unique_images[test_unique_indices]

    train_indices = full_dataset.labels_df[full_dataset.labels_df.iloc[:, 0].isin(train_imgs)].index.tolist()
    val_indices = full_dataset.labels_df[full_dataset.labels_df.iloc[:, 0].isin(val_imgs)].index.tolist()
    test_indices = full_dataset.labels_df[full_dataset.labels_df.iloc[:, 0].isin(test_imgs)].index.tolist()

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

if __name__ == '__main__':

    ABSOLUTE_PATH_TO_DATASET_ROOT = r'C:\Users\antonio\Desktop\dataset tesi'


    csv_file_path = os.path.join(ABSOLUTE_PATH_TO_DATASET_ROOT, 'balanced_regression.csv')

    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    print(f"Loading dataset from: {ABSOLUTE_PATH_TO_DATASET_ROOT}")
    image_dataset = MergedDataset(csv_file=csv_file_path,
                                       root_dir=ABSOLUTE_PATH_TO_DATASET_ROOT,
                                       transform=data_transform,
                                       classification=True)
    
    dataloader = DataLoader(image_dataset, 
                            batch_size=32, 
                            shuffle=True, 
                            num_workers=1,
                            pin_memory=False)
    
    for l, p in dataloader:
        print(p)
        break

    
    print(label_distribution(dataloader,1,labels_names=['age']))