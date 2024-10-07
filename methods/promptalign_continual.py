import numpy as np
import torch 
import os
from copy import deepcopy
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from utils.clip_tta_utils import compute_os_variance, accuracy, cal_auc_fpr, HM
from torch.nn import functional as F
from utils.registry import METHODS_REGISTRY


TPT_THRESHOLD = 0.1
ALIGN_THRESHOLD = 0.1
DISTR_LOSS_W = 100.0
visual_means = torch.load('weights/maple/ImgNet_vis_means.pt')
visual_vars = torch.load('weights/maple/ImgNet_vis_vars.pt')
ALIGN_LAYER_FROM = 0
ALIGN_LAYER_TO = 3

def select_confident_samples(logits, topTPT, topAlign):
    batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
    idxTPT = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * topTPT)]
    idxAlign = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * topAlign)]
    return logits[idxTPT], idxAlign


def avg_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True) # logits = outputs.log_softmax(dim=1) [N, 1000]
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0]) # avg_logits = logits.mean(0) [1, 1000]
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)


def distr_align_loss(out_feat, targ_feat, layers_from=0, layers_to=12, moments=5):
    '''
    A feature distibution alignment L1 loss between mean and variance of the features
    '''
    distr_loss = 0
    out_means, out_vars = out_feat
    targ_means, targ_vars = targ_feat
    transf_layers = layers_to
    for l in range(layers_from, transf_layers-1):
        out_mean, out_var = out_means[l], out_vars[l]
        targ_mean, targ_var = targ_means[l], targ_vars[l]
        distr_loss += 0.5 * F.l1_loss(out_mean, targ_mean) + 0.5 * F.l1_loss(out_var, targ_var)
    return distr_loss
    
    
def promptalign_test_time_tuning(model, inputs, optimizer, scaler):

    selected_idx = None
    DISTR_LOSS_W = 100.0
    for j in range(1):
        with torch.cuda.amp.autocast():
            output = model(inputs) 

            output, selected_idx = select_confident_samples(output, TPT_THRESHOLD, ALIGN_THRESHOLD)

            loss = avg_entropy(output)

            # Only selected indexes
            target_feat_distr = (visual_means, visual_vars)
            out_visual_mean = torch.cat([torch.mean(res.visual_feat[:, selected_idx, :], dim=1, keepdims=True).permute(1,0,2) for res in model.image_encoder.transformer.resblocks])
            out_visual_var = torch.cat([torch.mean(((res.visual_feat[:, selected_idx, :] - out_visual_mean[i, :, :].unsqueeze(0).permute(1,0,2))**2), dim=1, keepdims=True).permute(1,0,2) for i, res in enumerate(model.image_encoder.transformer.resblocks)])
            out_feat_distr = (out_visual_mean, out_visual_var)
        
            DISTR_LOSS_W = DISTR_LOSS_W / (ALIGN_LAYER_TO - ALIGN_LAYER_FROM)
            loss += DISTR_LOSS_W * distr_align_loss(out_feat_distr, target_feat_distr, 
                                        layers_from=ALIGN_LAYER_FROM, layers_to=ALIGN_LAYER_TO)
        
        optimizer.zero_grad()
        # compute gradient and do SGD step
        scaler.scale(loss).backward()
        # Unscales the gradients of optimizer's assigned params in-place
        scaler.step(optimizer)
        scaler.update()

    return model


@METHODS_REGISTRY.register()
def PromptAlignContinual(args, model, data_loader, classifiers=None):

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

    top1, top5, n = 0, 0, 0
    scores_q = []
    scores_length = args.N_scores

    model.set_prompt_inits() 
    for nm, param in model.named_parameters():
        if "prompt_learner" not in nm:
            param.requires_grad_(False)
            
    trainable_param = model.prompt_learner.parameters()
    # optimizer = torch.optim.AdamW(trainable_param, lr=4e-2)
    optimizer = torch.optim.SGD(trainable_param, lr=0.00001, momentum=0.9)
    print(optimizer)
    optim_state = deepcopy(optimizer.state_dict())
    scaler = torch.cuda.amp.GradScaler(init_scale=1000)

    model.eval()

    for i, (images, gt) in tqdm(enumerate(data_loader)):
        images = images[:-1]
        if isinstance(images,list):
            for k in range(len(images)):
                images[k] = images[k].cuda()
            image = images[0]
        else:
            image = image.cuda()
        images = torch.cat(images, dim=0)
        image, gt = image.cuda(), gt.cuda()
        ood_data['D'].append((gt<1000).item())
        ood_data['U'].append((gt>=1000).item())
        ood_data['gt_idx'].append(gt.item())
        
        #PromptAlign for continuous update: No reset
        with torch.no_grad():
            # model.reset()
        
            # TTA
            image_features = model.encode_image(image)
            image_features = image_features/image_features.norm(dim=-1, keepdim=True)

            tta_classifier = model.get_text_features()
            logits = image_features @ tta_classifier.T
            
        maxlogit_tta, pred_tta = logits.max(1)

        threshold_range = np.arange(0,1,0.01)
        ood_data['scores'].extend(maxlogit_tta.tolist())
        scores_q.extend(maxlogit_tta.tolist())
        scores_q = scores_q[-scores_length:]
        criterias = [compute_os_variance(np.array(scores_q), th) for th in threshold_range]
        best_thresh = threshold_range[np.argmin(criterias)]

        W_curr, S_curr = gt<1000, gt>=1000
        W_pred, S_pred = maxlogit_tta >= best_thresh, maxlogit_tta < best_thresh

        if W_pred[0].item():    
            # optimizer.load_state_dict(optim_state) #for continuous update
            model = promptalign_test_time_tuning(model, images, optimizer, scaler)


        # metrics
        n_samples['D_total'] += torch.sum(W_curr).item()
        n_samples['U_det'] += torch.sum(S_pred[S_curr]).item()
        n_samples['U_total'] += torch.sum(S_curr).item()

        with torch.no_grad():
            with torch.cuda.amp.autocast():                
                imf_norm = model.encode_image(image)
                imf_norm = imf_norm/imf_norm.norm(dim=-1, keepdim=True)
                tta_classifier = model.get_text_features().detach()
                logits_txt =  (imf_norm @ tta_classifier.T)
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

    return metrics_exp

