
import numpy as np
import torch

import warnings
from sklearn import metrics
from sklearn.metrics import roc_curve as Roc
from scipy import interpolate


warnings.filterwarnings("ignore")


def get_classifiers(model):
    classifiers = {}
    with torch.no_grad():
        classifiers['txt'] = model.get_text_features().detach()
    return classifiers


def get_ln_params(model):
    names, params = [], []
    for nm, p in model.named_parameters():
        if ('visual' in nm or 'image_encoder' in nm) and 'ln' in nm or 'bn' in nm:
            names.append(nm)
            params.append(p)
    return params

def compute_os_variance(os, th):
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
        return np.inf

    # find all pixels belonging to each class
    val_pixels1 = os[thresholded_os == 1]
    val_pixels0 = os[thresholded_os == 0]

    # compute variance of these classes
    var0 = np.var(val_pixels0) if len(val_pixels0) > 0 else 0
    var1 = np.var(val_pixels1) if len(val_pixels1) > 0 else 0

    return weight0 * var0 + weight1 * var1


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def HM(a,b):
    return 2*a*b/(a+b)
    
def cal_auc_fpr(ind_conf, ood_conf):
    conf = np.concatenate((ind_conf, ood_conf))
    ind_indicator = np.concatenate((np.ones_like(ind_conf), np.zeros_like(ood_conf)))
    auroc = metrics.roc_auc_score(ind_indicator, conf)
    fpr,tpr,thresh = Roc(ind_indicator, conf, pos_label=1)
    fpr = float(interpolate.interp1d(tpr, fpr)(0.95))
    return auroc, fpr

