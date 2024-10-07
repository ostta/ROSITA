import numpy as np
import torch 
import torch.nn as nn
import os
import json
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.clip_tta_utils import compute_os_variance, accuracy, cal_auc_fpr, HM, get_ln_params
from utils.registry import METHODS_REGISTRY


def normal_dist(x, mean, sd):
    prob_density = (np.pi*sd) * np.exp(-0.5*((x-mean)/sd)**2)
    return prob_density


def compute_os_variance_stats(os, th):
    """
    Parameters:
        os : score queue.
        th : Given threshold to separate desired and undesired class samples.

    Returns:
        float: Weighted variance at the given threshold th.
    """
    
    thresholded_os = np.zeros(os.shape)
    thresholded_os[os >= th] = 1

    # compute weights
    nb_pixels = os.size
    nb_pixels1 = np.count_nonzero(thresholded_os)
    weight1 = nb_pixels1 / nb_pixels
    weight0 = 1 - weight1

    # if one the classes is empty, eg all pixels are below or above the threshold, that threshold will not be considered
    # in the search for the best threshold
    if weight1 == 0 or weight0 == 0:
        return np.inf, {'mu0': 0, 'mu1': 1, 'var0': 0, 'var1': 0}

    # find all pixels belonging to each class
    val_pixels1 = os[thresholded_os == 1]
    val_pixels0 = os[thresholded_os == 0]

    # compute mean of these classes
    mu0 = np.mean(val_pixels0) if len(val_pixels0) > 0 else 0
    mu1 = np.mean(val_pixels1) if len(val_pixels1) > 0 else 0

    # compute variance of these classes
    var0 = np.var(val_pixels0) if len(val_pixels0) > 0 else 0
    var1 = np.var(val_pixels1) if len(val_pixels1) > 0 else 0

    stats = {'mu0': mu0, 'mu1': mu1, 'var0': var0, 'var1': var1}
    return weight0 * var0 + weight1 * var1, stats


@METHODS_REGISTRY.register()
def ROSITA(args, model, data_loader, classifiers):

    tta_method = args.tta_method
    
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

    scores_q = []
    scores_length = args.N_scores
    queue_length = args.N_s

    classifier = classifiers[args.classifier_type]
    C = classifier.shape[0]

    ln_params = get_ln_params(model)
    optimizer = torch.optim.SGD(ln_params, lr=args.tta_lr)
    log_file.write(str(args))
    log_file.write(str(optimizer))
    log_file.write(f'\n Length of dataloader: {len(data_loader)}')

    proto_bank = {'D': [], 'U': []}

    

    for i, (image, gt) in tqdm(enumerate(data_loader)):
        if isinstance(image,list):
            images = torch.cat(image[:2], dim=0).cuda()
            image, image_aug = image[0].cuda(), image[1].cuda()
        else:
            image = image.cuda()
        image, gt = image.cuda(), gt.cuda()
        ood_data['D'].append((gt<1000).item())
        ood_data['U'].append((gt>=1000).item())
        ood_data['gt_idx'].append(gt.item())

        # TTA
        image_features_raw_all = model.encode_image(images)
        image_features_all = image_features_raw_all/image_features_raw_all.norm(dim=-1, keepdim=True)


        image_features_raw = image_features_raw_all[0].unsqueeze(0)
        image_features = image_features_all[0].unsqueeze(0)

        logits_all = image_features_all @ classifier.T
        logits, logits_aug = logits_all[0].unsqueeze(0), logits_all[1].unsqueeze(0)
        maxlogit_tta, pred_tta = logits.max(1)

        threshold_range = np.arange(0,1,0.01)
        ood_data['scores'].extend(maxlogit_tta.tolist())
        scores_q.extend(maxlogit_tta.tolist())
        scores_q = scores_q[-scores_length:]
        criterias = [compute_os_variance(np.array(scores_q), th) for th in threshold_range]
        best_thresh = threshold_range[np.argmin(criterias)]
        _, stats = compute_os_variance_stats(np.array(scores_q), best_thresh)

        W_curr, S_curr = gt<1000, gt>=1000
        W_pred, S_pred = maxlogit_tta >= best_thresh, maxlogit_tta < best_thresh

        pl = pred_tta[0].item()

        W_sel = W_pred * (maxlogit_tta > stats['mu1'])

        loss = 0
        if W_pred[0].item() and maxlogit_tta.item()>stats['mu1']:
            proto_bank['D'].append(image_features_raw.detach())
            proto_bank['D'] = proto_bank['D'][-C * args.k_p:]

        if S_pred[0].item() and maxlogit_tta.item()<stats['mu0']:
            proto_bank['U'].append(image_features_raw.detach())
            proto_bank['U'] = proto_bank['U'][-queue_length:]

        if args.loss_pl and W_sel:

            loss += nn.CrossEntropyLoss()(logits[W_sel], pred_tta[W_sel])
            loss += nn.CrossEntropyLoss()(logits_aug[W_sel], pred_tta[W_sel])

        knn_acc= -1

        if args.loss_simclr and len(proto_bank['D'])>args.k_p and len(proto_bank['U'])>args.k_p:
            if maxlogit_tta.item()>stats['mu1'] or maxlogit_tta.item()<stats['mu0']:
            
                if W_pred[0].item():
                    n_p = args.k_p
                    pos_features = torch.vstack(proto_bank['D']) 
                    neg_features = torch.vstack(proto_bank['U'])

                    pos_features = pos_features/pos_features.norm(dim=-1, keepdim=True)
                    neg_features = neg_features/neg_features.norm(dim=-1, keepdim=True)

                    pos_sim = image_features @ pos_features.T
                    topk_pos_sim, topk_pos_idx = torch.topk(pos_sim, k=n_p, dim=-1, largest=True)

                    topk_pos_idx = topk_pos_idx.squeeze()
                    positive_features = pos_features[topk_pos_idx]
                    if n_p ==1:
                        positive_features = positive_features.unsqueeze(0)
                    pos_logits = positive_features @ classifier.T
                    pos_preds = pos_logits.argmax(1)
                    mask = pl==pos_preds
                    
                    knn_acc= torch.sum(mask).cpu().item()

                    if i%100==0:
                        status_log = f'iter:{i}; pos_feature_bank_len:{pos_features.shape[0]}'
                        log_file.write(status_log)
                        print(status_log)

                    neg_sim = image_features @ neg_features.T
                    topk_neg_sim, _ = torch.topk(neg_sim, k=args.k_n, dim=-1, largest=True) 

                    pos_sim_k = topk_pos_sim.T
                    neg_sim_k = topk_neg_sim.expand(n_p, -1)

                    l_simclr = -pos_sim_k.squeeze() + torch.log(torch.exp(neg_sim_k).sum(1))
                    l_simclr = l_simclr * mask
                    l_simclr = torch.mean(l_simclr)

                    loss += l_simclr * args.alpha

                elif S_pred[0].item(): 
                    pos_features = torch.vstack(proto_bank['U'])
                    neg_features = torch.vstack(proto_bank['D']) 

                    pos_features = pos_features/pos_features.norm(dim=-1, keepdim=True)
                    neg_features = neg_features/neg_features.norm(dim=-1, keepdim=True)

                    pos_sim = image_features @ pos_features.T
                    topk_pos_sim, topk_pos_idx = torch.topk(pos_sim, k=args.k_p, dim=-1, largest=True)
                    
                    neg_sim = image_features @ neg_features.T
                    topk_neg_sim, _ = torch.topk(neg_sim, k=args.k_n, dim=-1, largest=True) 

                    pos_sim_k = topk_pos_sim.T
                    neg_sim_k = topk_neg_sim.expand(args.k_p, -1)

                    l_simclr = -pos_sim_k.squeeze() + torch.log(torch.exp(neg_sim_k).sum(1))
                    if W_pred[0].item(): 
                        l_simclr = l_simclr * mask

                    l_simclr = torch.mean(l_simclr)

                    loss += l_simclr * args.alpha

        if loss:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # metrics
        # n_samples['W_det'] += torch.sum(W_pred[W_curr]).item()
        n_samples['D_total'] += torch.sum(W_curr).item()
        n_samples['U_det'] += torch.sum(S_pred[S_curr]).item()
        n_samples['U_total'] += torch.sum(S_curr).item()

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                imf_norm = model.encode_image(image)
                imf_norm = imf_norm/imf_norm.norm(dim=-1, keepdim=True)
                logits_txt =  (imf_norm @ classifier.T)
                scores_txt = (logits_txt * 100).softmax(1)
                _, pred_tta = torch.max(scores_txt, dim=1)

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
