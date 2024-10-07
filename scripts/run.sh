#!/bin/bash
MODEL=$1
TTA_METHOD=$2
GPU_ID=$3


CUDA_VISIBLE_DEVICES=${GPU_ID} python owtta.py --desired cifar10OOD --undesired MNIST    --model ${MODEL} --tta_method ${TTA_METHOD}
CUDA_VISIBLE_DEVICES=${GPU_ID} python owtta.py --desired cifar10OOD --undesired SVHN     --model ${MODEL} --tta_method ${TTA_METHOD}
CUDA_VISIBLE_DEVICES=${GPU_ID} python owtta.py --desired cifar10OOD --undesired cifar100 --model ${MODEL} --tta_method ${TTA_METHOD}
CUDA_VISIBLE_DEVICES=${GPU_ID} python owtta.py --desired cifar10OOD --undesired Tiny     --model ${MODEL} --tta_method ${TTA_METHOD}
CUDA_VISIBLE_DEVICES=${GPU_ID} python owtta.py --desired cifar100OOD --undesired MNIST   --model ${MODEL} --tta_method ${TTA_METHOD}
CUDA_VISIBLE_DEVICES=${GPU_ID} python owtta.py --desired cifar100OOD --undesired SVHN    --model ${MODEL} --tta_method ${TTA_METHOD}
CUDA_VISIBLE_DEVICES=${GPU_ID} python owtta.py --desired cifar100OOD --undesired cifar10 --model ${MODEL} --tta_method ${TTA_METHOD}
CUDA_VISIBLE_DEVICES=${GPU_ID} python owtta.py --desired cifar100OOD --undesired Tiny    --model ${MODEL} --tta_method ${TTA_METHOD}
CUDA_VISIBLE_DEVICES=${GPU_ID} python owtta.py --tesize 50000 --desired ImagenetCOOD --undesired MNIST  --model ${MODEL} --tta_method ${TTA_METHOD}
CUDA_VISIBLE_DEVICES=${GPU_ID} python owtta.py --tesize 50000 --desired ImagenetCOOD --undesired SVHN   --model ${MODEL} --tta_method ${TTA_METHOD}
CUDA_VISIBLE_DEVICES=${GPU_ID} python owtta.py --tesize 30000 --desired ImagenetROOD --undesired MNIST  --model ${MODEL} --tta_method ${TTA_METHOD}
CUDA_VISIBLE_DEVICES=${GPU_ID} python owtta.py --tesize 30000 --desired ImagenetROOD --undesired SVHN   --model ${MODEL} --tta_method ${TTA_METHOD}
CUDA_VISIBLE_DEVICES=${GPU_ID} python owtta.py --tesize 50000 --desired VisdaOOD --undesired MNIST      --model ${MODEL} --tta_method ${TTA_METHOD}
CUDA_VISIBLE_DEVICES=${GPU_ID} python owtta.py --tesize 50000 --desired VisdaOOD --undesired SVHN       --model ${MODEL} --tta_method ${TTA_METHOD}