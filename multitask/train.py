import argparse, os, sys, torch
from pathlib import Path
import os
import sys
from dotenv import load_dotenv
load_dotenv()
REPO_PATH = os.getenv("REPO_PATH")
if REPO_PATH:
    sys.path.append(REPO_PATH)

from trainer import Trainer
from config.task_config import MTL_TASK_CONFIG
from config.task_config import Task, MTLConfig
from multitask.ordinalLoss import OrdinalRegressionLoss
# 'google/Siglip2-base-patch16-224'
# 'PE-Core-B16-224'
# 'PE-Core-T16-384' 



from datetime import datetime

def log_config_to_file(config, args):
    """
    Creates the output directory and saves all configuration details to info.txt.
    """
    # 1. Define the output directory from the config
    output_dir = config.output_folder
    
    # 2. Ensure the directory exists. The parents=True flag creates any missing parent folders.
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 3. Define the full path for the info file
    info_file_path = output_dir / 'info.txt'
    
    print(f"Saving configuration to {info_file_path}...")

    # 4. Open the file in write mode ('w') and write the information
    with open(info_file_path, 'w') as f:
        f.write("="*50 + "\n")
        f.write("Experiment Configuration Details\n")
        f.write(f"Run executed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*50 + "\n\n")

        # --- Write Command Line Arguments ---
        f.write("--- Command Line Arguments (args) ---\n")
        # vars(args) converts the argparse.Namespace object to a dictionary
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")
        f.write("\n")

        # --- Write MTL_TASK_CONFIG details ---
        f.write("--- MTL Task Configuration ---\n")
        f.write(f"output_folder: {config.output_folder}\n")
        f.write(f"dataset_root: {config.dataset_root}\n")
        f.write(f"train_csv: {config.train_csv}\n")
        f.write(f"val_csv: {config.val_csv}\n")
        f.write(f"test_csv: {config.test_csv}\n")
        f.write(f"use_uncertainty_weighting: {config.use_uncertainty_weighting}\n")
        f.write(f"use_grad_norm: {config.use_grad_norm}\n")
        f.write(f"grad_norm_alpha: {config.grad_norm_alpha}\n")
        f.write(f"use_lora: {config.use_lora}\n")
        f.write(f"use_dwa: {config.use_dwa}\n\n")
        
        # --- Write Individual Task details ---
        f.write("--- Individual Tasks ---\n")
        for i, task in enumerate(config.tasks, 1):
            f.write(f"  Task {i}: {task.name}\n")
            f.write(f"    class_labels: {task.class_labels}\n")
            f.write(f"    criterion: {task.criterion.__name__}\n")
            f.write(f"    weight: {task.weight}\n")
            if hasattr(task, 'use_weighted_loss'):
                 f.write(f"    use_weighted_loss: {task.use_weighted_loss}\n")
            f.write("\n")
            
    print("Configuration saved successfully.")


supported_models = [
    'google/siglip2-base-patch16-224',
    'PE-Core-B16-224',
    'PE-Core-T16-384',
    'google/siglip2-large-patch16-256',
    'PE-Core-L14-336'
]   

CHOSEN = 4
def main():
    parser = argparse.ArgumentParser(description="Train and validate attention probes for different tasks.")
    parser.add_argument('--version', type=str, default=supported_models[CHOSEN], help='Backbone model version.')
    parser.add_argument('--ckpt_path', type=str, help='Path to the backbone checkpoint. Only for PE models.')
    parser.add_argument('--resume_from_ckpt', type=str, help='Path to a probe checkpoint to resume training from.')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs.')
    # parser.add_argument('--dataset_root', type=str, default=os.getenv("DATASET_ROOT"), help='Root directory of the dataset images.')
    # parser.add_argument('--csv_path_gender', type=str,default='/user/asessa/dataset tesi/gender_labels_cropped.csv' ,help='Path to the CSV file with labels for training split.')
    # parser.add_argument('--csv_path_emotions', type=str, default='/user/asessa/dataset tesi/emotion_labels_cropped.csv', help='Path to the CSV file with labels for training split.')
    # parser.add_argument('--csv_path_age', type=str, help='Path to the CSV file with labels for training split.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training.')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for optimizer.')
    parser.add_argument('--moe', type=bool, default=False, help='Use task-aware mixture of experts.')
    parser.add_argument('--k_probes', type=bool, default=False, help='Use k-task specific probes to produce three distinct task-embeddings for each classifier head')
    parser.add_argument('--testing', type=bool, default=False, help='Skip straight to testing')
    parser.add_argument('--load_pt', type=bool, default=False, help='Load pre-trained heads from .pt files')
    parser.add_argument('--initial_ul' , type=int, default=0, help='How many layers to unfreeze at the start of training. If 0, only attn_pool layer is unfrozen. If -1 only the heads are trained (no mt learning)')
    parser.add_argument('--deeper_classification_heads', type=bool, default=False, help='2 hidden layers in the classfiication heads')
    parser.add_argument('--name',type=str,default='AGE_LORA_PE')
    parser.add_argument('--tasks',type=str,default='age')
    args = parser.parse_args()
    print('Start training with the following args:')
    print(args)
    pre_traiend_heads_siglip = {
        'age' : '/user/asessa/tesi/probing/experiments/age_classification/Siglip2-base-patch16-224/ckpt_lp/lp_age_Siglip2-base-patch16-224_47.pt',
        'gender' : '/user/asessa/tesi/probing/experiments/gender_classification/Siglip2-base-patch16-224/ckpt_lp/lp_gender_Siglip2-base-patch16-224_2.pt',
        'emotion' : '/user/asessa/tesi/probing/experiments/emotion_classification/Siglip2-base-patch16-224/ckpt_lp/lp_emotion_Siglip2-base-patch16-224_49.pt'
    } 

    pre_trained_head_pe = {
        'age' : '/user/asessa/tesi/probing/experiments/age_classification/PE-Core-B16-224/ckpt_lp/lp_age_PE-Core-B16-224_58.pt',
        'gender' : '/user/asessa/tesi/probing/experiments/gender_classification/PE-Core-B16-224/ckpt_lp/lp_gender_PE-Core-B16-224_4.pt',
        'emotion': '/user/asessa/tesi/probing/experiments/emotion_classification/PE-Core-B16-224/ckpt_lp/lp_emotion_PE-Core-B16-224_58.pt'
    }

    """
    csvs with VGG:
    /user/asessa/test_folder/val/validation.csv ->validation
    /user/asessa/test_folder/test/test.csv      ->test
    /user/asessa/test_folder/train/train.csv    ->train

    csvs without VGG: 
    /user/asessa/dataset tesi/small_train.csv
    /user/asessa/dataset tesi/mtl_test.csv
    /user/asessa/dataset tesi/fairface_adiance_raf_affect.csv ***
    /user/asessa/dataset tesi/adience_data_filtered_cropped.csv
    /user/asessa/dataset tesi/small_train_age_gender.csv

    # TEST SETS!
    /user/asessa/dataset tesi/test_sets/fairface-test.csv
    /user/asessa/dataset tesi/test_sets/raf-db-test.csv
    /user/asessa/dataset tesi/test_sets/utk-test.csv
    /user/asessa/dataset tesi/datasets_with_standard_labels/VggFace2/test/vgg_labels_test.csv
    /user/asessa/dataset tesi/labels_imdb_test.csv

    WITH VGG:
    /user/asessa/dataset tesi/GENDER_TRAIN_VGG+.csv
    /user/asessa/dataset tesi/VALIDATION_VGG.csv
    /user/asessa/dataset tesi/datasets_with_standard_labels/VggFace2/test/labels_test_vgg_fixed.csv #TEST
    /user/asessa/dataset tesi/TRAIN_AGE_GENDER.csv (fairface, vgg, adiance, raf-db, lagenda, imdb)

    debug:
    /user/asessa/dataset tesi/fast_testing.csv


    /user/asessa/dataset tesi/train_final.csv (ff, raf-db, adiance, lagenda, imdb) RICORDA, aggiungi AffectNet se serve ~376k
    /user/asessa/dataset tesi/train_age.csv (ff, adiance, lagenda, imdb) RICORDA, aggiungi AffectNet se serve ~376k
    /user/asessa/dataset tesi/validation.csv (imdb, ff, lagenda) ~63k
    
    fairface -> /user/asessa/dataset tesi/small_train_age_gender.csv
    """
    # /user/asessa/dataset tesi/test_sets/fairface-test.csv
    # /user/asessa/dataset tesi/test_sets/utk-test.csv

    if args.tasks == 'all':
        MTL_TASK_CONFIG = MTLConfig(
            tasks=[
                Task(name='Age', class_labels=["0-2","3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"], criterion=torch.nn.CrossEntropyLoss, weight=1.0, use_weighted_loss=True),
                Task(name='Gender', class_labels=["Male", "Female"], criterion=torch.nn.CrossEntropyLoss, weight=1.0),
                #Task(name='Emotion', class_labels=["Surprise", "Fear", "Disgust", "Happy", "Sad", "Angry", "Neutral"], criterion=torch.nn.CrossEntropyLoss, weight=1.0, use_weighted_loss=True)
            ],
                output_folder=Path(f'./GENDER_LP_TEST_PEVITL'),
                dataset_root=Path("/user/asessa/dataset tesi/"), 
                train_csv=Path("/user/asessa/dataset tesi/train_final.csv"),
                val_csv=Path("/user/asessa/dataset tesi/validation.csv"),
                test_csv=Path("/user/asessa/dataset tesi/labels_imdb_test.csv"),
                use_uncertainty_weighting=False,
                use_grad_norm=False,
                grad_norm_alpha=1.5,
                use_lora=False, # modify SA in Pe ViT if using lora with pe 
                use_dwa=False
            )
    if args.tasks == 'gender':
        MTL_TASK_CONFIG = MTLConfig(
            tasks=[
                #Task(name='Age', class_labels=["0-2","3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"], criterion=torch.nn.CrossEntropyLoss, weight=1.0, use_weighted_loss=True),
                Task(name='Gender', class_labels=["Male", "Female"], criterion=torch.nn.CrossEntropyLoss, weight=1.0),
                #Task(name='Emotion', class_labels=["Surprise", "Fear", "Disgust", "Happy", "Sad", "Angry", "Neutral"], criterion=torch.nn.CrossEntropyLoss, weight=1.0, use_weighted_loss=True)
            ],
                output_folder=Path(f'./{args.name}'),
                dataset_root=Path("/user/asessa/dataset tesi/"), 
                train_csv=Path("/user/asessa/dataset tesi/train_final.csv"),
                val_csv=Path("/user/asessa/dataset tesi/validation.csv"),
                test_csv=Path("/user/asessa/dataset tesi/datasets_with_standard_labels/VggFace2/test/vgg_labels_test.csv"),
                use_uncertainty_weighting=False,
                use_grad_norm=False,
                grad_norm_alpha=1.5,
                use_lora=True, # modify SA in Pe ViT if using lora with pe 
                use_dwa=False
            )
    if args.tasks == 'age':
        MTL_TASK_CONFIG = MTLConfig(
            tasks=[
                Task(name='Age', class_labels=["0-2","3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"], criterion=torch.nn.CrossEntropyLoss, weight=1.0, use_weighted_loss=True),
                #Task(name='Gender', class_labels=["Male", "Female"], criterion=torch.nn.CrossEntropyLoss, weight=1.0),
                #Task(name='Emotion', class_labels=["Surprise", "Fear", "Disgust", "Happy", "Sad", "Angry", "Neutral"], criterion=torch.nn.CrossEntropyLoss, weight=1.0, use_weighted_loss=True)
            ],
                output_folder=Path(f'./{args.name}'),
                dataset_root=Path("/user/asessa/dataset tesi/"), 
                train_csv=Path("/user/asessa/dataset tesi/train_final.csv"),
                val_csv=Path("/user/asessa/dataset tesi/validation.csv"),
                test_csv=Path("/user/asessa/dataset tesi/test_sets/utk-test.csv"),
                use_uncertainty_weighting=False,
                use_grad_norm=False,
                grad_norm_alpha=1.5,
                use_lora=True, # modify SA in Pe ViT if using lora with pe 
                use_dwa=False
            )

    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting.")
        sys.exit(1)

    task_config = MTL_TASK_CONFIG
    
    if args.testing:
        trainer = Trainer(config=task_config, args=args, offset= 0)
        trainer.test(ckpt_path=args.resume_from_ckpt)
        exit()

    try:
        trainer = Trainer(config=task_config, args=args, offset= 1 if args.tasks == 'gender' else 0)
        log_config_to_file(MTL_TASK_CONFIG, args)
        if args.load_pt:
            if 'google' in args.version:
                trainer.load_heads(pre_traiend_heads_siglip)
            else:
                trainer.load_heads(pre_trained_head_pe)
        trainer.train()
        # trainer.test()
    finally:
        print("Executing final cleanup...")
        if trainer is not None:
            trainer.cleanup()


if __name__ == '__main__':
    main()
