python train.py --task age --probe_type linear --epochs 70 --version 'PE-Core-B16-224'
python train.py --task age --probe_type linear --epochs 70

python train.py --task emotion --probe_type linear --epochs 70 --version 'PE-Core-B16-224'
python train.py --task emotion --probe_type linear --epochs 70

python train.py --task gender --probe_type linear --epochs 30 --version 'PE-Core-B16-224'
python train.py --task gender --probe_type linear --epochs 30


python train.py --task age --probe_type attention --epochs 70 --version 'PE-Core-B16-224'
python train.py --task age --probe_type attention --epochs 70

python train.py --task emotion --probe_type attention --epochs 70 --version 'PE-Core-B16-224'
python train.py --task emotion --probe_type attention --epochs 70

python train.py --task gender --probe_type attention --epochs 30 --version 'PE-Core-B16-224'
python train.py --task gender --probe_type attention --epochs 30