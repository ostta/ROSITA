import math
import operator
import numpy as np
import torch 
import torch.nn as nn
import os
import json
from copy import deepcopy
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.clip_tta_utils import compute_os_variance, accuracy, cal_auc_fpr, HM, get_ln_params
from torch.nn import functional as F
from utils.registry import METHODS_REGISTRY


TPT_THRESHOLD = 0.1
ALIGN_THRESHOLD = 0.1


def select_confident_samples(logits, topTPT, topAlign):
    batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
    idxTPT = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * topTPT)]
    idxAlign = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * topAlign)]
    return logits[idxTPT], idxAlign


def avg_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True) 
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0]) 
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)


def softmax_entropy(x):
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def get_entropy(loss, clip_weights):
    max_entropy = math.log2(clip_weights.size(1))
    return float(loss / max_entropy)


def get_clip_logits(images, clip_model, clip_weights):
    with torch.no_grad():
        image_features = clip_model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        clip_logits = 100. * image_features @ clip_weights

        if image_features.size(0) > 1:
            batch_entropy = softmax_entropy(clip_logits)
            selected_idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * 0.1)]
            output = clip_logits[selected_idx]
            image_features = image_features[selected_idx].mean(0).unsqueeze(0)
            clip_logits = output.mean(0).unsqueeze(0)

            loss = avg_entropy(output)
            prob_map = output.softmax(1).mean(0).unsqueeze(0)
            pred = int(output.mean(0).unsqueeze(0).topk(1, 1, True, True)[1].t())
        else:
            loss = softmax_entropy(clip_logits)
            prob_map = clip_logits.softmax(1)
            pred = int(clip_logits.topk(1, 1, True, True)[1].t()[0])

        return image_features, clip_logits, loss, prob_map, pred
    

def update_cache(cache, pred, features_loss, shot_capacity, include_prob_map=False):
    """Update cache with new features and loss, maintaining the maximum shot capacity."""
    with torch.no_grad():
        item = features_loss if not include_prob_map else features_loss[:2] + [features_loss[2]]
        if pred in cache:
            if len(cache[pred]) < shot_capacity:
                cache[pred].append(item)
            elif features_loss[1] < cache[pred][-1][1]:
                cache[pred][-1] = item
            cache[pred] = sorted(cache[pred], key=operator.itemgetter(1))
        else:
            cache[pred] = [item]


def compute_cache_logits(image_features, cache, alpha, beta, clip_weights, neg_mask_thresholds=None):
    """Compute logits using positive/negative cache."""
    with torch.no_grad():
        cache_keys = []
        cache_values = []
        for class_index in sorted(cache.keys()):
            for item in cache[class_index]:
                cache_keys.append(item[0])
                if neg_mask_thresholds:
                    cache_values.append(item[2])
                else:
                    cache_values.append(class_index)

        cache_keys = torch.cat(cache_keys, dim=0).permute(1, 0)
        if neg_mask_thresholds:
            cache_values = torch.cat(cache_values, dim=0)
            cache_values = (((cache_values > neg_mask_thresholds[0]) & (cache_values < neg_mask_thresholds[1])).type(torch.int8)).cuda().half()
        else:
            cache_values = (F.one_hot(torch.Tensor(cache_values).to(torch.int64), num_classes=clip_weights.size(1))).cuda().half()

        affinity = (image_features @ cache_keys).half()
        cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
        return alpha * cache_logits



@METHODS_REGISTRY.register()
def TDA(args, model, data_loader, classifiers=None):

    tta_method = args.tta_method
    tta_classifier = classifiers[args.classifier_type]
    
    log_dir_path = os.path.join(args.out_dir, args.model, args.desired, args.undesired, tta_method)
    os.makedirs(log_dir_path, exist_ok=True)
    log_file = open(f'{log_dir_path}/{tta_method}.txt', 'w')

    n_samples= {}
    n_samples['ALL'] = 0 
    n_samples['D'] = 0 
    n_samples['U_det'] = 0
    n_samples['D_total'] = 0
    n_samples['U_total'] = 0


    metrics_exp = {'Method':tta_method , 'AUC':0, 'FPR95':0, 'ACC_D':0, 'ACC_U':0, 'ACC_HM':0}
    ood_data = {'D': [], 'U': [], 'gt_idx': [], 'scores': []}

    top1, top5, n = 0, 0, 0
    scores_q = []
    scores_length = args.N_scores

    model.set_prompt_inits() 
    for nm, param in model.named_parameters():
        if "prompt_learner" not in nm:
            param.requires_grad_(False)
    

    model.eval()

    pos_cfg = {'enabled':True, 'shot_capacity':3, 'alpha':1.0, 'beta':8.0}
    neg_cfg = {'enabled':True, 'shot_capacity':2, 'alpha':0.117, 'beta':1.0, 'entropy_threshold': {'lower':0.2, 'upper':1.0}, 'mask_threshold': {'lower':0.03, 'upper': 1.0}}

    pos_cache, neg_cache = {}, {}

    pos_enabled, neg_enabled = pos_cfg['enabled'], neg_cfg['enabled']
    if pos_enabled:
        pos_params = {k: pos_cfg[k] for k in ['shot_capacity', 'alpha', 'beta']}
    if neg_enabled:
        neg_params = {k: neg_cfg[k] for k in ['shot_capacity', 'alpha', 'beta', 'entropy_threshold', 'mask_threshold']}


    for i, (images, gt) in tqdm(enumerate(data_loader)):
        image = images[0]
        image, gt = image.cuda(), gt.cuda()
        ood_data['D'].append((gt<1000).item())
        ood_data['U'].append((gt>=1000).item())
        ood_data['gt_idx'].append(gt.item())
        
        #TDA
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                model.reset()

                image_features, clip_logits, loss, prob_map, pred = get_clip_logits(image ,model, tta_classifier.T)
                prop_entropy = get_entropy(loss, tta_classifier.T)

                if pos_enabled:
                    update_cache(pos_cache, pred, [image_features, loss], pos_params['shot_capacity'])

                if neg_enabled and neg_params['entropy_threshold']['lower'] < prop_entropy < neg_params['entropy_threshold']['upper']:
                    update_cache(neg_cache, pred, [image_features, loss, prob_map], neg_params['shot_capacity'], True)

                logits = clip_logits.clone()
                if pos_enabled and pos_cache:
                    logits += compute_cache_logits(image_features, pos_cache, pos_params['alpha'], pos_params['beta'], tta_classifier.T)
                if neg_enabled and neg_cache:
                    logits -= compute_cache_logits(image_features, neg_cache, neg_params['alpha'], neg_params['beta'], tta_classifier.T, (neg_params['mask_threshold']['lower'], neg_params['mask_threshold']['upper']))
                
                maxlogit_tta, pred_tta = logits.max(1)

        threshold_range = np.arange(0,100,1)
        ood_data['scores'].extend(maxlogit_tta.tolist())
        scores_q.extend(maxlogit_tta.tolist())
        scores_q = scores_q[-scores_length:]
        criterias = [compute_os_variance(np.array(scores_q), th) for th in threshold_range]
        best_thresh = threshold_range[np.argmin(criterias)]

        W_curr, S_curr = gt<1000, gt>=1000
        W_pred, S_pred = maxlogit_tta >= best_thresh, maxlogit_tta < best_thresh

        # metrics
        n_samples['D_total'] += torch.sum(W_curr).item()
        n_samples['U_det'] += torch.sum(S_pred[S_curr]).item()
        n_samples['U_total'] += torch.sum(S_curr).item()

        if W_pred[0].item():
            n_samples['D'] += torch.sum(gt[gt<1000]==pred_tta[gt<1000]).item()
        n_samples['ALL'] += torch.sum(gt[gt<1000]==pred_tta[gt<1000]).item()

        if (i+1) %1000 == 0:
            acc_w = n_samples['D']/n_samples['D_total']
            acc_s = n_samples['U_det']/n_samples['U_total']
            acc_hm = HM(acc_w, acc_s)
            status_log = f'\nStep {i}:   ACC_D: {acc_w}; ACC_U: {acc_s}; ACC_HM: {acc_hm}'
            print(status_log)
            log_file.write(status_log)

    metrics_exp['ACC_D'] = n_samples['D']/n_samples['D_total']
    metrics_exp['ACC_U'] = n_samples['U_det']/n_samples['U_total']

    ood_data['scores'] = np.array(ood_data['scores'])
    metrics_exp['AUC'], metrics_exp['FPR95'] = cal_auc_fpr(ood_data['scores'][ood_data['D']], ood_data['scores'][ood_data['U']])
    metrics_exp['ACC_HM'] = HM(metrics_exp['ACC_D'], metrics_exp['ACC_U'])

    print(f'\n\n')
    print(args.desired, args.undesired, tta_method)
    print(f'Metrics: {metrics_exp}\n')

    df_metrics = pd.DataFrame([metrics_exp])
    df_metrics.to_csv(f'{log_dir_path}/{args.tta_method}_results.csv', index=False)  

    plt.hist(ood_data['scores'][ood_data['D']], bins=threshold_range, label='Desired', alpha=0.5)
    plt.hist(ood_data['scores'][ood_data['U']], bins=threshold_range, label='Undesired', alpha=0.5)
    plt.legend()
    plt.savefig(f'{log_dir_path}/{args.tta_method}_scores_hist.jpg')

    return metrics_exp


