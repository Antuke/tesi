#python train.py --task age --probe_type linear --epochs 70 --version 'PE-Core-B16-224'
#python train.py --task age --probe_type linear --epochs 70

#python train.py --task emotion --probe_type linear --epochs 70 --version 'PE-Core-B16-224'
#python train.py --task emotion --probe_type linear --epochs 70

#python train.py --task gender --probe_type linear --epochs 10 --version 'PE-Core-B16-224'
#python train.py --task gender --probe_type linear --epochs 10


#python train.py --task age --probe_type attention --epochs 70 --version 'PE-Core-B16-224'
#python train.py --task age --probe_type attention --epochs 70

python train.py --task emotion --probe_type attention --epochs 70 --version 'PE-Core-B16-224'
python train.py --task emotion --probe_type attention --epochs 70

python train.py --task gender --probe_type attention --epochs 10 --version 'PE-Core-B16-224'
python train.py --task gender --probe_type attention --epochs 10