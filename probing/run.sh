
# Common settings
DATASET_ROOT="/user/asessa/dataset tesi/"
GENDER_CSV_PATH="/user/asessa/dataset tesi/balanced_classification.csv"
AGE_CSV_PATH="/user/asessa/dataset tesi/age_labels_classification.csv"
EMOTION_CSV_PATH="/user/asessa/dataset tesi/emotion_labels_cropped.csv"
CKPT_PATH="C:\Users\antonio\Desktop\perception_models\ckpt\PE-Core-B16-224.pt"


# 'PE-Core-B16-224'


python train.py \
    --task emotion \
    --version 'google/Siglip2-base-patch16-224' \
    --dataset_root "$DATASET_ROOT" \
    --csv_path "$EMOTION_CSV_PATH" \
    --probe_type "attention"  \
    --epochs 70 