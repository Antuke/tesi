
# Common settings
DATASET_ROOT="/user/asessa/dataset tesi/"
GENDER_CSV_PATH="/user/asessa/dataset tesi/gender_labels_cropped.csv"
AGE_CSV_PATH="/user/asessa/dataset tesi/age_labels_cropped.csv"
EMOTION_CSV_PATH="/user/asessa/dataset tesi/emotion_labels_cropped.csv"


# PE LINEAR
python train.py \
    --task gender \
    --version 'PE-Core-B16-224' \
   --dataset_root "$DATASET_ROOT" \
    --csv_path "$GENDER_CSV_PATH" \
    --probe_type "linear"  \
    --epochs 30 \
    --resume_from_ckpt "/user/asessa/tesi/probing/gender_outputs/ckpt/lp_gender_PE-Core-B16-224_30.pt"



python train.py \
    --task age_classification \
    --version 'PE-Core-B16-224' \
    --dataset_root "$DATASET_ROOT" \
    --csv_path "$AGE_CSV_PATH" \
    --probe_type "linear"  \
    --epochs 70 \



python train.py \
    --task emotion \
    --version 'PE-Core-B16-224' \
    --dataset_root "$DATASET_ROOT" \
    --csv_path "$EMOTION_CSV_PATH" \
    --probe_type "linear"  \
    --epochs 70 \


# PE ATTENTION

python train.py \
    --task gender \
    --version 'PE-Core-B16-224' \
    --dataset_root "$DATASET_ROOT" \
    --csv_path "$GENDER_CSV_PATH" \
    --probe_type "attention"  \
    --epochs 30 \


python train.py \
    --task age_classification \
    --version 'PE-Core-B16-224' \
    --dataset_root "$DATASET_ROOT" \
    --csv_path "$AGE_CSV_PATH" \
    --probe_type "attention"  \
    --epochs 70 \


python train.py \
    --task emotion \
    --version 'PE-Core-B16-224' \
    --dataset_root "$DATASET_ROOT" \
    --csv_path "$EMOTION_CSV_PATH" \
    --probe_type "attention"  \
    --epochs 70 \




# SIGLIP LINEAR

python train.py \
    --task gender \
    --version 'google/Siglip2-base-patch16-224' \
    --dataset_root "$DATASET_ROOT" \
    --csv_path "$GENDER_CSV_PATH" \
    --probe_type "linear"  \
    --epochs 30 \


python train.py \
    --task age_classification \
    --version 'google/Siglip2-base-patch16-224' \
    --dataset_root "$DATASET_ROOT" \
    --csv_path "$AGE_CSV_PATH" \
    --probe_type "linear"  \
    --epochs 70 \


python train.py \
    --task emotion \
    --version 'google/Siglip2-base-patch16-224' \
    --dataset_root "$DATASET_ROOT" \
    --csv_path "$EMOTION_CSV_PATH" \
    --probe_type "linear"  \
    --epochs 70 \


# SIGLIP ATTENTION

python train.py \
    --task gender \
    --version 'google/Siglip2-base-patch16-224' \
    --dataset_root "$DATASET_ROOT" \
    --csv_path "$GENDER_CSV_PATH" \
    --probe_type "attention"  \
    --epochs 30 \


python train.py \
    --task age_classification \
    --version 'google/Siglip2-base-patch16-224' \
    --dataset_root "$DATASET_ROOT" \
    --csv_path "$AGE_CSV_PATH" \
    --probe_type "attention"  \
    --epochs 70 \


python train.py \
    --task emotion \
    --version 'google/Siglip2-base-patch16-224' \
    --dataset_root "$DATASET_ROOT" \
    --csv_path "$EMOTION_CSV_PATH" \
    --probe_type "attention"  \
    --epochs 70 \