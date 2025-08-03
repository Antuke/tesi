
# Common settings
DATASET_ROOT="/user/asessa/dataset tesi/"
GENDER_CSV_PATH="/user/asessa/dataset tesi/gender_labels_cropped.csv"
AGE_CSV_PATH="/user/asessa/dataset tesi/age_labels_cropped.csv"
EMOTION_CSV_PATH="/user/asessa/dataset tesi/emotion_labels_cropped.csv"






exec python train.py \
    --task gender \
    --version 'PE-Core-T16-384' \
    --dataset_root "$DATASET_ROOT" \
    --csv_path "$GENDER_CSV_PATH" \
    --probe_type "attention"  \
    --epochs 70 \


exec python train.py \
    --task age_classification \
    --version 'PE-Core-T16-384' \
    --dataset_root "$DATASET_ROOT" \
    --csv_path "$AGE_CSV_PATH" \
    --probe_type "attention"  \
    --epochs 70 \


exec python train.py \
    --task emotion \
    --version 'PE-Core-T16-384' \
    --dataset_root "$DATASET_ROOT" \
    --csv_path "$EMOTION_CSV_PATH" \
    --probe_type "attention"  \
    --epochs 70 \