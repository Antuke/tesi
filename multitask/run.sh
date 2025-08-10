DATASET_ROOT="C:\Users\antonio\Desktop\dataset tesi"
DEFAULT_CSV_PATH="C:\Users\antonio\Desktop\dataset tesi\balanced_classification.csv"
CKPT_PATH="../ckpt/PE-Core-B16-224.pt"


# google/Siglip2-base-patch16-224


# pe
python train.py \
    --version 'PE-Core-B16-224' \
    --dataset_root "$DATASET_ROOT" \
    --csv_path "$DEFAULT_CSV_PATH" \
    --epochs 5 \



python train.py --task emotion --version PE-Core-B16-224 --resume_from_ckpt /user/asessa/tesi/probing/experiments/emotion_classification/ckpt/lp_emotion_PE-Core-B16-224_30.pt