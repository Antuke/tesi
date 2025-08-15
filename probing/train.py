import argparse, os, sys, torch
import os
import sys
from dotenv import load_dotenv
# Environment and Path Setup
load_dotenv()
REPO_PATH = os.getenv("REPO_PATH")
if REPO_PATH:
    sys.path.append(REPO_PATH)

from trainer import Trainer
from config.task_config import TASK_REGISTRY

# 'google/Siglip2-base-patch16-224'
# 'PE-Core-B16-224'


def main():
    parser = argparse.ArgumentParser(description="Train and validate attention probes for different tasks.")
    parser.add_argument('--task', type=str, required=True, choices=TASK_REGISTRY.keys(), help='Task to perform.')
    parser.add_argument('--version', type=str, default='google/Siglip2-base-patch16-224', help='Backbone model version.')
    parser.add_argument('--ckpt_path', type=str, help='Path to the backbone checkpoint. Only for PE models.')
    parser.add_argument('--resume_from_ckpt', type=str, help='Path to a probe checkpoint to resume training from.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')
    # parser.add_argument('--dataset_root', type=str, default=os.getenv("DATASET_ROOT"), help='Root directory of the dataset images.')
    # parser.add_argument('--csv_path', type=str, help='Path to the CSV file with labels for training split.')
    parser.add_argument('--probe_type', type=str, default='linear', choices=['attention', 'linear'], help='Type of probing.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer.')
    parser.add_argument('--test', type=bool, default=True, help='Skip to testing')

    args = parser.parse_args()
    #if args.version == 'PE-Core-T16-384':
    #    torch.multiprocessing.set_start_method('spawn', force=True)

    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting.")
        sys.exit(1)

    task_config = TASK_REGISTRY[args.task]
    if True:
        print('only testing!')
        trainer = Trainer(config=task_config, args=args)
        trainer.test(ckpt_path='/user/asessa/tesi/multitask/experiment/ckpt/mtl_Siglip2_ul4_10.pt')
        exit()
    try:
        trainer = Trainer(config=task_config, args=args)
        trainer.train()
        trainer.test()
    finally:
        print("Executing final cleanup...")
        if trainer is not None:
            trainer.cleanup()


if __name__ == '__main__':
    main()