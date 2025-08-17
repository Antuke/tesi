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

# 'google/Siglip2-base-patch16-224'
# 'PE-Core-B16-224'

def main():
    parser = argparse.ArgumentParser(description="Train and validate attention probes for different tasks.")
    parser.add_argument('--version', type=str, default='PE-Core-B16-224', help='Backbone model version.')
    parser.add_argument('--ckpt_path', type=str, help='Path to the backbone checkpoint. Only for PE models.')
    parser.add_argument('--resume_from_ckpt', type=str, help='Path to a probe checkpoint to resume training from.')
    parser.add_argument('--epochs', type=int, default=280, help='Number of training epochs.')
    # parser.add_argument('--dataset_root', type=str, default=os.getenv("DATASET_ROOT"), help='Root directory of the dataset images.')
    # parser.add_argument('--csv_path_gender', type=str,default='/user/asessa/dataset tesi/gender_labels_cropped.csv' ,help='Path to the CSV file with labels for training split.')
    # parser.add_argument('--csv_path_emotions', type=str, default='/user/asessa/dataset tesi/emotion_labels_cropped.csv', help='Path to the CSV file with labels for training split.')
    # parser.add_argument('--csv_path_age', type=str, help='Path to the CSV file with labels for training split.')
    parser.add_argument('--num_layers_to_unfreeze', type=int, default='0',  help='How many layers to unfreeze.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training.')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for optimizer.')
    parser.add_argument('--moe', type=bool, default=True, help='Use task-aware mixture of experts.')
    parser.add_argument('--k_probes', type=bool, default=False, help='Use k-task specific probes to produce three distinct task-embeddings for each classifier head')
    parser.add_argument('--testing', type=bool, default=False, help='Skip straight to testing')
    args = parser.parse_args()
    
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
    """

    MTL_TASK_CONFIG = MTLConfig(
    tasks=[
        Task(name='Age', class_labels=["0-2","3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"], criterion=torch.nn.CrossEntropyLoss, weight=1.0, use_weighted_loss=True),
        Task(name='Gender', class_labels=["Male", "Female"], criterion=torch.nn.CrossEntropyLoss, weight=1.0),
        Task(name='Emotion', class_labels=["Surprise", "Fear", "Disgust", "Happy", "Sad", "Angry", "Neutral"], criterion=torch.nn.CrossEntropyLoss, weight=1.0, use_weighted_loss=True)
    ],
        output_folder=Path('./pe_outputs_uncertainty_weighting_big'),
        dataset_root=Path("/user/asessa/dataset tesi/"), 
        train_csv=Path("/user/asessa/test_folder/train/train.csv"),
        val_csv=Path("/user/asessa/dataset tesi/mtl_test.csv"),
        test_csv=Path("/user/asessa/dataset tesi/mtl_test.csv"),
        use_uncertainty_weighting=True,
        use_grad_norm=False,
        grad_norm_alpha=1.5
    )
    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting.")
        sys.exit(1)

    task_config = MTL_TASK_CONFIG
    
    if args.testing:
        print('!!!TESTING!!!')
        trainer = Trainer(config=task_config, args=args)
        trainer.test(ckpt_path="/user/asessa/tesi/multitask/pe_outputs_uncertainty_weighting/ckpt/mtl_PE-Core-B16-224_ul0_1.pt")
        exit()

    try:
        #trainer = Trainer(config=task_config, args=args)
        #trainer.load_heads(pre_traiend_heads)
        #trainer.train()
        trainer = Trainer(config=task_config, args=args)
        if 'google' in args.version:
            trainer.load_heads(pre_traiend_heads_siglip)
        else:
            trainer.load_heads(pre_trained_head_pe)
        trainer.train()
        trainer.test()
    finally:
        print("Executing final cleanup...")
        if trainer is not None:
            trainer.cleanup()


if __name__ == '__main__':
    main()