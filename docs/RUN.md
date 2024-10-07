### Main Experiments

To reproduce the main results of ROSITA on all datasets, run

```
sh scripts/run.sh clip ROSITA 0
sh scripts/run.sh maple ROSITA 0
```

The baselines can be run using the same script by choosing the VLM backbone and the TTA method as follows

```
sh scripts/run.sh [clip/maple] [ZSEval/ROSITA/TPT/TPTContinual/PromptAlign/PromptAlignContinual/PC/TDA] GPU_ID
```

To run a specific experiment, select the desired, undesired class datasets, base VLM and the TTA method to evaluate from the following list.

```
python owtta.py --desired [cifar10OOD/cifar100OOD/ImageNetCOOD/ImageNetROOD/VisDAOOD] --undesired [MNIST/SVHN/Tiny/cifar10/cifar100] --model [clip/maple] --tta_method [ZSEval/ROSITA/TPT/TPTContinual/PromptAlign/PromptAlignContinual/PC/TDA]
```

For e.g., to run PromptAlign using MaPLe as the VLM, CIFAR-10C as desired class and Tiny ImageNet as undesired class dataset, the following command should be used.

```
python owtta.py --desired cifar10OOD --undesired Tiny --model maple --tta_method ROSITA
```

### Supported desired and Undesired combinations

```
python owtta.py --desired cifar10OOD --undesired [MNIST/SVHN/Tiny/cifar100] --model [clip/maple] --tta_method ROSITA
python owtta.py --desired cifar100OOD --undesired [MNIST/SVHN/Tiny/cifar10] --model [clip/maple] --tta_method ROSITA
python owtta.py --desired ImageNetCOOD --undesired [MNIST/SVHN] --model [clip/maple] --tta_method ROSITA
python owtta.py --desired ImageNetROOD --undesired [MNIST/SVHN] --model [clip/maple] --tta_method ROSITA
python owtta.py --desired VisDAOOD --undesired [MNIST/SVHN] --model [clip/maple] --tta_method ROSITA
```

### Supported VLM and Method combinations

```
python owtta.py --desired cifar10OOD --undesired Tiny --model clip --tta_method [ZSEval/ROSITA/TPT/TPTContinual/PC/TDA]
python owtta.py --desired cifar10OOD --undesired Tiny --model maple --tta_method [ZSEval/ROSITA/TPT/TPTContinual/PromptAlign/PromptAlignContinual]
```
