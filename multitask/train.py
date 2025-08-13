import argparse, os, sys, torch
import os
import sys
from dotenv import load_dotenv
load_dotenv()
REPO_PATH = os.getenv("REPO_PATH")
if REPO_PATH:
    sys.path.append(REPO_PATH)

from trainer import Trainer
from config.task_config import MTL_TASK_CONFIG


# 'google/Siglip2-base-patch16-224'
# 'PE-Core-B16-224'

def main():
    parser = argparse.ArgumentParser(description="Train and validate attention probes for different tasks.")
    parser.add_argument('--version', type=str, default='google/Siglip2-base-patch16-224', help='Backbone model version.')
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
    args = parser.parse_args()

    pre_traiend_heads = {
        'age' : '/user/asessa/tesi/probing/experiments/age_classification/Siglip2-base-patch16-224/ckpt_lp/lp_age_Siglip2-base-patch16-224_47.pt',
        'gender' : '/user/asessa/tesi/probing/experiments/gender_classification/Siglip2-base-patch16-224/ckpt_lp/lp_gender_Siglip2-base-patch16-224_2.pt',
        'emotion' : '/user/asessa/tesi/probing/experiments/emotion_classification/Siglip2-base-patch16-224/ckpt_lp/lp_emotion_Siglip2-base-patch16-224_49.pt'
    } 

    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting.")
        sys.exit(1)

    task_config = MTL_TASK_CONFIG
    try:
        #trainer = Trainer(config=task_config, args=args)
        #trainer.load_heads(pre_traiend_heads)
        #trainer.train()
        trainer = Trainer(config=task_config, args=args)
        trainer.load_heads(pre_traiend_heads)
        trainer.train()
        trainer.test()
    finally:
        print("Executing final cleanup...")
        if trainer is not None:
            trainer.cleanup()


if __name__ == '__main__':
    main()