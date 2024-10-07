import os
import sys
import random
import torch
import torchvision
from PIL import Image
from typing import Sequence, Callable, Optional

# ID Datasets
class CIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        return
    
    def __getitem__(self, index: int):
        image, target = super().__getitem__(index)
        if type(image) == list:
            image.append(index)
        else:
            image = [image, index]
        return image, target


class CIFAR100(torchvision.datasets.CIFAR100):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        return
    
    def __getitem__(self, index: int):
        image, target = super().__getitem__(index)
        if type(image) == list:
            image.append(index)
        else:
            image = [image, index]
        return image, target


class ImageNetR(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None,list=True,ratio=1, tesize=10000):
        self.Train = True
        self.list=list
        self.root_dir = os.path.join(root, 'imagenet_r')
        self.transform = transform
        self.train_dir = os.path.join(self.root_dir, "data")
        self.val_dir = os.path.join(self.root_dir, "data")
        self.ratio = ratio
                
        words_file = os.path.join(self.root_dir, "words_wordnet.txt")
        wnids_file = os.path.join(self.root_dir, "wnids.txt")

        self.set_nids = set()
        self.class_list = []
        with open(wnids_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                self.set_nids.add(entry.strip("\n"))
                self.class_list.append(entry.strip("\n"))

        self.class_to_label = {}
        with open(words_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                words = entry.split("\t")
                if words[0] in self.set_nids:
                    self.class_to_label[words[0]] = (words[1].strip("\n").split(","))[0]
        self.classnames = self.class_to_label.values()

        if (self.Train):
            self._create_class_idx_dict_train()
        else:
            self._create_class_idx_dict_val()

        self._make_dataset(self.Train, tesize=tesize)


    def _create_class_idx_dict_train(self):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(self.train_dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(self.train_dir) if os.path.isdir(os.path.join(train_dir, d))]
        classes = sorted(classes)
        num_images = 0
        temp=[]
        for i in range(20):
            temp.append(0)
        for root, dirs, files in os.walk(self.train_dir):
            for f in files:
                if f.endswith(".jpg"):
                    num_images = num_images + 1

        self.len_dataset = num_images;

        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}

    def _create_class_idx_dict_val(self):
        val_image_dir = os.path.join(self.val_dir, "images")
        if sys.version_info >= (3, 5):
            images = [d.name for d in os.scandir(val_image_dir) if d.is_file()]
        else:
            images = [d for d in os.listdir(val_image_dir) if os.path.isfile(os.path.join(train_dir, d))]
        val_annotations_file = os.path.join(self.val_dir, "val_annotations.txt")
        self.val_img_to_class = {}
        set_of_classes = set()
        with open(val_annotations_file, 'r') as fo:
            entry = fo.readlines()
            for data in entry:
                words = data.split("\t")
                if words[1] in self.class_list:
                    self.val_img_to_class[words[0]] = words[1]
                    set_of_classes.add(words[1])

        self.len_dataset = len(list(self.val_img_to_class.keys()))
        classes = sorted(list(set_of_classes))
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}
        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}

    def _make_dataset(self, Train=True, tesize=10000):
        self.images = []
        if Train:
            img_root_dir = self.train_dir
            list_of_dirs = [target for target in self.class_to_tgt_idx.keys()]
        else:
            img_root_dir = self.val_dir
            list_of_dirs = ["images"]
        temp=[]
        for i in range(20):
            temp.append(0)

        for root, _, files in os.walk(self.train_dir):
            for fname in sorted(files):
                if (fname.endswith(".jpg")): 
                    path = os.path.join(root, fname)
                    item = (path, self.class_to_tgt_idx[root.split('/')[-1]])
                    self.images.append(item)
        random.shuffle(self.images)
        self.images = self.images[:tesize]
        self.len_dataset = len(self.images)
        print('len',len(self.images))

    def return_label(self, idx):
        return [self.class_to_label[self.tgt_idx_to_class[i.item()]] for i in idx]

    def __len__(self):
        return int(self.len_dataset*self.ratio)

    def __getitem__(self, idx:int):
        img_path, tgt = self.images[idx]
        with open(img_path, 'rb') as f:
            sample = Image.open(img_path)
            sample = sample.convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        index = idx
        if self.list:
            if type(sample) == list:
                sample.append(index)
            else:
                sample = [sample, index]

        return sample, tgt


class VISDA(torch.utils.data.Dataset):
    def __init__(self, root: str, label_files: Sequence[str], transform: Optional[Callable] = None, tesize=10000):
        self.image_root = root
        self.label_files = label_files
        self.transform = transform

        self.samples = self.build_index(label_file=label_files, tesize=tesize) 

    def build_index(self, label_file, tesize):
        """Build a list of <image path, class label, domain name> items.
        Input:
            label_file: Path to the file containing the image label pairs
        Returns:
            item_list: A list of <image path, class label> items.
        """
        with open(label_file, "r") as file:
            tmp_items = [line.strip().split() for line in file if line]
        random.shuffle(tmp_items)
        tmp_items = tmp_items[:tesize]

        item_list = []
        for img_file, label in tmp_items:
            img_file = f"{os.sep}".join(img_file.split("/"))
            img_path = os.path.join(self.image_root, img_file)
            item_list.append((img_path, int(label)))

        return item_list

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        if type(image) == list:
            image.append(idx)
        else:
            image = [image, idx]

        return image, label


# Undesired class Datasets
class CIFAR100_openset(torchvision.datasets.CIFAR100):
    def __init__(self, tesize=10000, ratio=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data, self.targets = self.data[:int(tesize*ratio)], self.targets[:int(tesize*ratio)]
        return
    
    def __getitem__(self, index: int):
        image, target = super().__getitem__(index)
        target = target + 1000
        if type(image) == list:
            image.append(index)
        else:
            image = [image, index]
        return image, target


class CIFAR10_openset(torchvision.datasets.CIFAR10):
    def __init__(self, tesize=10000, ratio=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data, self.targets = self.data[:int(tesize*ratio)], self.targets[:int(tesize*ratio)]
        return
    
    def __getitem__(self, index: int):
        image, target = super().__getitem__(index)
        target = target + 1000
        if type(image) == list:
            image.append(index)
        else:
            image = [image, index]
        return image, target


class MNIST_openset(torchvision.datasets.MNIST):
    def __init__(self, *args, tesize=10000, ratio = 1 , **kwargs):
        super().__init__(*args, **kwargs)
        self.data, self.targets = self.data[:int(tesize*ratio)], self.targets[:int(tesize*ratio)]
        return
    
    def __getitem__(self, index: int):
        image, target = super().__getitem__(index)
        target = target + 1000
        if type(image) == list:
            image.append(index)
        else:
            image = [image, index]
        return image, target


class SVHN_openset(torchvision.datasets.SVHN):
    def __init__(self, *args, tesize=10000, ratio = 1 , **kwargs):
        super().__init__(*args, **kwargs)
        self.data, self.labels = self.data[:int(tesize*ratio)], self.labels[:int(tesize*ratio)]
        return
    
    def __getitem__(self, index: int):
        image, target = super().__getitem__(index)
        target = target + 1000
        if type(image) == list:
            image.append(index)
        else:
            image = [image, index]
        return image, target


class TinyImageNet_OOD_nonoverlap(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None,list=True,ratio=1):
        self.Train = train
        self.list=list
        self.root_dir = root
        self.transform = transform
        self.train_dir = os.path.join(self.root_dir, "train")
        self.val_dir = os.path.join(self.root_dir, "val")
        self.ratio = ratio
        
        self.class_list = ['n03544143', 'n03255030', 'n04532106', 'n02669723', 'n02321529', 'n02423022', 'n03854065', 'n02509815', 'n04133789', 'n03970156', 'n01882714', 'n04023962', 'n01768244', 'n04596742', 'n03447447', 'n03617480', 'n07720875', 'n02125311', 'n02793495', 'n04532670']

        if (self.Train):
            self._create_class_idx_dict_train()
        else:
            self._create_class_idx_dict_val()

        self._make_dataset(self.Train)

        words_file = os.path.join(self.root_dir, "words.txt")
        wnids_file = os.path.join(self.root_dir, "wnids.txt")

        self.set_nids = set()

        with open(wnids_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                if entry.strip("\n") in self.class_list:
                    self.set_nids.add(entry.strip("\n"))

        self.class_to_label = {}
        with open(words_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                words = entry.split("\t")
                if words[0] in self.set_nids:
                    self.class_to_label[words[0]] = (words[1].strip("\n").split(","))[0]

    def _create_class_idx_dict_train(self):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(self.train_dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(self.train_dir) if os.path.isdir(os.path.join(train_dir, d))]
        classes = sorted(classes)
        num_images = 0
        temp=[]
        for i in range(20):
            temp.append(0)
        for root, dirs, files in os.walk(self.train_dir):
            for f in files:
                if f.endswith(".JPEG") and f.split("_")[0] in self.class_list:
                    for i in range(len(self.class_list)):
                        if f.split("_")[0] == self.class_list[i]:
                            
                            
                            if temp[i] < 500:
                                temp[i]+=1
                                num_images = num_images + 1
                            break
        self.len_dataset = num_images;

        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}

    def _create_class_idx_dict_val(self):
        val_image_dir = os.path.join(self.val_dir, "images")
        if sys.version_info >= (3, 5):
            images = [d.name for d in os.scandir(val_image_dir) if d.is_file()]
        else:
            images = [d for d in os.listdir(val_image_dir) if os.path.isfile(os.path.join(train_dir, d))]
        val_annotations_file = os.path.join(self.val_dir, "val_annotations.txt")
        self.val_img_to_class = {}
        set_of_classes = set()
        with open(val_annotations_file, 'r') as fo:
            entry = fo.readlines()
            for data in entry:
                words = data.split("\t")
                if words[1] in self.class_list:
                    self.val_img_to_class[words[0]] = words[1]
                    set_of_classes.add(words[1])

        self.len_dataset = len(list(self.val_img_to_class.keys()))
        classes = sorted(list(set_of_classes))
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}
        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}

    def _make_dataset(self, Train=True):
        self.images = []
        if Train:
            img_root_dir = self.train_dir
            list_of_dirs = [target for target in self.class_to_tgt_idx.keys()]
        else:
            img_root_dir = self.val_dir
            list_of_dirs = ["images"]
        temp=[]
        for i in range(20):
            temp.append(0)
        for tgt in list_of_dirs:
            dirs = os.path.join(img_root_dir, tgt)
            if not os.path.isdir(dirs):
                continue

            for root, _, files in sorted(os.walk(dirs)):
                for fname in sorted(files):
                    if (fname.endswith(".JPEG"))and fname.split("_")[0] in self.class_list:
                        path = os.path.join(root, fname)
                        if Train:
                            item = (path, self.class_to_tgt_idx[tgt])
                        else:
                            item = (path, self.class_to_tgt_idx[self.val_img_to_class[fname]])
                        for i in range(len(self.class_list)):
                            if fname.split("_")[0] == self.class_list[i]:
                                temp[i]+=1
                                
                                if temp[i] <= 500:
                                    self.images.append(item)
        print('len',len(self.images))

    def return_label(self, idx):
        return [self.class_to_label[self.tgt_idx_to_class[i.item()]] for i in idx]

    def __len__(self):
        return int(self.len_dataset*self.ratio)

    def __getitem__(self, idx:int):
        img_path, tgt = self.images[idx]
        tgt+=1000
        with open(img_path, 'rb') as f:
            sample = Image.open(img_path)
            sample = sample.convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        index = idx
        if self.list:
            if type(sample) == list:
                sample.append(index)
            else:
                sample = [sample, index]

        return sample, tgt


class DomainNet(torch.utils.data.Dataset):
    def __init__(self, root: str, label_files: Sequence[str], transform: Optional[Callable] = None, tesize=10000):
        self.image_root = root
        self.label_files = label_files
        self.transform = transform

        self.samples = self.build_index(label_file=label_files, tesize=tesize) 

    def build_index(self, label_file, tesize):
        """Build a list of <image path, class label, domain name> items.
        Input:
            label_file: Path to the file containing the image label pairs
        Returns:
            item_list: A list of <image path, class label> items.
        """
        with open(label_file, "r") as file:
            tmp_items = [line.strip().split() for line in file if line]
        random.shuffle(tmp_items)
        tmp_items = tmp_items[:tesize]

        item_list = []
        for img_file, label in tmp_items:
            img_file = f"{os.sep}".join(img_file.split("/"))
            img_path = os.path.join(self.image_root, img_file)
            item_list.append((img_path, int(label)))

        return item_list

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        if type(image) == list:
            image.append(idx)
        else:
            image = [image, idx]

        return image, label