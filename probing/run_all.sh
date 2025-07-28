
# Common settings
DATASET_ROOT="C:\Users\antonio\Desktop\dataset tesi"
DEFAULT_CSV_PATH="C:\Users\antonio\Desktop\dataset tesi\merged_labels.csv"
AGE_CSV_PATH="C:\Users\antonio\Desktop\dataset tesi\age_labels_classification.csv"
EMOTION_CSV_PATH="C:\Users\antonio\Desktop\dataset tesi\emotions_labels.csv"
CKPT_PATH="C:\Users\antonio\Desktop\perception_models\ckpt\PE-Core-B16-224.pt"




# LINEAR PROBING
# Task: age
python probing.py \
    --task age_classification \
    --version 'PE-Core-B16-224' \
    --ckpt_path $CKPT_PATH \
    --dataset_root "$DATASET_ROOT" \
    --csv_path "$AGE_CSV_PATH" \
    --probe_type "attention" \

python probing.py \
    --task age_classification \
    --version 'google/Siglip2-base-patch16-224' \
    --dataset_root "$DATASET_ROOT" \
    --csv_path "$AGE_CSV_PATH" \
    --probe_type "attention" \




# Task: emotion
python probing.py \
    --task emotion \
    --version 'PE-Core-B16-224' \
    --ckpt_path $CKPT_PATH \
    --dataset_root "$DATASET_ROOT" \
    --csv_path "$EMOTION_CSV_PATH" \
    --probe_type "attention"

python probing.py \
    --task emotion \
    --version 'google/Siglip2-base-patch16-224' \
    --dataset_root "$DATASET_ROOT" \
    --csv_path "$EMOTION_CSV_PATH" \
    --probe_type "attention"



# Task: gender
python probing.py \
    --task gender \
    --version 'PE-Core-B16-224' \
    --ckpt_path $CKPT_PATH \
    --dataset_root "$DATASET_ROOT" \
    --csv_path "$DEFAULT_CSV_PATH" \
    --probe_type "attention"

python probing.py \
    --task gender \
    --version 'google/Siglip2-base-patch16-224' \
    --dataset_root "$DATASET_ROOT" \
    --csv_path "$DEFAULT_CSV_PATH" \
    --probe_type "attention"


# LINEAR PROBING
# Task: age
python probing.py \
    --task age_classification \
    --version 'PE-Core-B16-224' \
    --ckpt_path $CKPT_PATH \
    --dataset_root "$DATASET_ROOT" \
    --csv_path "$AGE_CSV_PATH" \
    --probe_type "linear"

python probing.py \
    --task age_classification \
    --version 'google/Siglip2-base-patch16-224' \
    --dataset_root "$DATASET_ROOT" \
    --csv_path "$AGE_CSV_PATH" \
    --probe_type "linear"

# Task: emotion
python probing.py \
    --task emotion \
    --version 'PE-Core-B16-224' \
    --ckpt_path $CKPT_PATH \
    --dataset_root "$DATASET_ROOT" \
    --csv_path "$EMOTION_CSV_PATH" \
    --probe_type "linear"

python probing.py \
    --task emotion \
    --version 'google/Siglip2-base-patch16-224' \
    --dataset_root "$DATASET_ROOT" \
    --csv_path "$EMOTION_CSV_PATH" \
    --probe_type "linear"



# Task: gender
python probing.py \
    --task gender \
    --version 'PE-Core-B16-224' \
    --ckpt_path $CKPT_PATH \
    --dataset_root "$DATASET_ROOT" \
    --csv_path "$DEFAULT_CSV_PATH" \
    --probe_type "linear"

python probing.py \
    --task gender \
    --version 'google/Siglip2-base-patch16-224' \
    --dataset_root "$DATASET_ROOT" \
    --csv_path "$DEFAULT_CSV_PATH" \
    --probe_type "linear"