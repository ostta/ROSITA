import argparse
import random
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC


from clip import clip
from models.maple import CustomCLIP_MaPLe
from models.clip import CustomCLIP_CLIP

from utils.data_utils import prepare_ood_test_data, AugMixAugmenter
from utils.clip_tta_utils import get_classifiers

from methods import rosita, zseval, tpt, tpt_continual, promptalign, promptalign_continual, tda, protocluster

from utils.registry import get_method



def load_model_to_cpu(args):
    url = clip._MODELS[args.clip_arch]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    if args.model == 'clip' or args.model == 'coop':
        design_details = {"trainer": 'CoOp',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0}

    elif args.model == 'maple':
        design_details = {"trainer": 'PromptAlign',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0,
                      "maple_length": 2}

    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


def get_preprocess_transforms(args):
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                        std=[0.26862954, 0.26130258, 0.27577711])

    preprocess = transforms.Compose([
        transforms.Resize(224, interpolation=BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,])
    
    if args.tta_method in ['TPT','PromptAlign', 'TPTContinual', 'PromptAlignContinual']:
        base_transform = transforms.Compose([
            transforms.Resize(224, interpolation=BICUBIC),
            transforms.CenterCrop(224)])
        preprocess = transforms.Compose([
                    transforms.ToTensor(),
                    normalize])
        preprocess = AugMixAugmenter(base_transform, preprocess, n_views=args.n_views-1, 
                                                augmix=False)

    if args.tta_method in ['ROSITA', 'PC']:
        base_transform = transforms.Compose([
            transforms.Resize(224, interpolation=BICUBIC),
            transforms.CenterCrop(224)])
        preprocess = transforms.Compose([
                    transforms.ToTensor(),
                    normalize])
        preprocess = AugMixAugmenter(base_transform, preprocess, n_views=1, 
                                                augmix=False)

    return preprocess


def get_model(args, classnames):

    clip_model = load_model_to_cpu(args)
    if args.model == 'clip':
        model = CustomCLIP_CLIP(classnames, clip_model)

    elif args.model == 'maple':
        model = CustomCLIP_MaPLe(classnames, clip_model)

        checkpoint = torch.load('weights/maple/model.pth.tar-2')
        state_dict = checkpoint["state_dict"]
        epoch = checkpoint["epoch"]

        # Ignore fixed token vectors
        if "prompt_learner.token_prefix" in state_dict:
            del state_dict["prompt_learner.token_prefix"]

        if "prompt_learner.token_suffix" in state_dict:
            del state_dict["prompt_learner.token_suffix"]

        model.load_state_dict(state_dict, strict=False)

    model.cuda()
    
    return model


parser = argparse.ArgumentParser()
parser.add_argument('--clip_arch', default='ViT-B/16')
parser.add_argument('--desired', default='cifar10OOD')
parser.add_argument('--undesired', default='MNIST') 
parser.add_argument('--strong_ratio', default=1, type=float)
parser.add_argument('--dataroot', default="./data", help='path to dataset')
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--n_views', default=64, type=int)
parser.add_argument('--workers', default=4, type=int)
parser.add_argument('--out_dir', default='./logs/', help='folder to output log')
parser.add_argument('--level', default=5, type=int)
parser.add_argument('--N_scores', default=512, type=int, help='scores length')
parser.add_argument('--N_s', default=64, type=int, help='queue length')
parser.add_argument('--corruption', default='snow')
parser.add_argument('--model', default='clip', help='VLM backbone: clip/maple')
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--tta_method', default='ZSEval', type=str)
parser.add_argument('--alpha', default=0.5, type=float)
parser.add_argument('--classifier_type', default='txt', type=str)
parser.add_argument('--k_p', default=5, type=int)
parser.add_argument('--k_n', default=5, type=int)
parser.add_argument('--loss_pl', default=1, type=int)
parser.add_argument('--loss_simclr', default=1, type=int)
parser.add_argument('--tesize', default=10000, type=int)
parser.add_argument('--param_group', default='ln', type=str)
parser.add_argument('--tta_lr', default=0.001, type=float)
parser.add_argument('--opt', default='SGD', type=str)





# ----------- Args and Dataloader ------------

if __name__ == "__main__":
    args = parser.parse_args()

    print(args)
    print('\n')

    if args.tta_method in ['PromptAlign', 'PromptAlignContinual']:
        assert args.model =='maple', "PromptAlign and PromptAlignContinual is supported with MaPLe only."
    
    if args.desired in ['ImagenetCOOD', 'ImagenetROOD', 'VisdaOOD']:
        assert args.undesired in ['MNIST', 'SVHN'], f"Only MNIST and SVHN are supported undesired class datasets with {args.desired} to avoid overlap of classes"

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    preprocess = get_preprocess_transforms(args)

    data_dict, test_set, test_loader = prepare_ood_test_data(args, preprocess)
    
    model = get_model(args, data_dict['ID_classes'])

    desired_classifiers = get_classifiers(model)

    method = get_method(args.tta_method)

    result_metrics = method(args, model, test_loader, desired_classifiers)

    print('\n\n\n')
