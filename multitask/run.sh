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



