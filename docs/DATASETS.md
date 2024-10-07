## Desired class datasets

- CIFAR-10C
- CIFAR-100C
- ImageNet-C
- ImageNet-R

## Undesired class datasets

- MNIST
- SVHN
- Tiny ImageNet
- CIFAR-10C
- CIFAR-100C

## Dataset Preparation

```
export DATADIR=/data/cifar
mkdir -p ${DATADIR} && cd ${DATADIR}
```

### CIFAR-10C

```
wget -O CIFAR-10-C.tar https://zenodo.org/record/2535967/files/CIFAR-10-C.tar?download=1
tar -xvf CIFAR-10-C.tar
```

### CIFAR-100C

```
wget -O CIFAR-100-C.tar https://zenodo.org/record/3555552/files/CIFAR-100-C.tar?download=1
tar -xvf CIFAR-100-C.tar
```

### Tiny ImageNet

```
wget -O tiny-imagenet-200.zip http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip
```

### ImageNet-R

```
wget -O imagenet-r.tar https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar
tar -xvf imagenet-r.tar
```

### ImageNet-C

For `snow` corruption, download [weather.tar](https://zenodo.org/records/2235448/files/weather.tar?download=1) (Source: [ImageNet-C](https://zenodo.org/records/2235448#.Yj2RO_co_mF)). Extarct and place it under the folder `data/ImageNet-C`. Ensuring the accessibilty of `data/ImageNet-C/snow/5` folder suffices to reproduce the main results.

To run on any other corruption types, download the required corruption of ImageNet-C from [here](https://zenodo.org/records/2235448#.Yj2RO_co_mF). Extract and place it under the folder `data/ImageNet-C`. Change the `--corruption` argument accordingly.

### VisDA

```
mkdir visda-2017 && cd visda-2017
wget http://csr.bu.edu/ftp/visda17/clf/validation.tar
tar xvf validation.tar
```
