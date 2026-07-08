import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import sklearn.metrics as sk
import clip_w_local


def print_measures(log, auroc, aupr, fpr, method_name='Ours', recall_level=0.95):
    if log is None:
        print('FPR{:d}:\t\t\t{:.2f}'.format(int(100 * recall_level), 100 * fpr))
        print('AUROC: \t\t\t{:.2f}'.format(100 * auroc))
        print('AUPR:  \t\t\t{:.2f}'.format(100 * aupr))
    else:
        log.debug('\t\t\t\t' + method_name)
        log.debug('  FPR{:d} AUROC AUPR'.format(int(100*recall_level)))
        log.debug('& {:.2f} & {:.2f} & {:.2f}'.format(100*fpr, 100*auroc, 100*aupr))


def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out


def fpr_and_fdr_at_recall(y_true, y_score, recall_level=0.95, pos_label=None):
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                     np.array_equal(classes, [-1, 1]) or
                     np.array_equal(classes, [0]) or
                     np.array_equal(classes, [-1]) or
                     np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)      # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true)))   # , fps[cutoff]/(fps[cutoff] + tps[cutoff])


def get_measures(_pos, _neg, recall_level=0.95):
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1

    auroc = sk.roc_auc_score(labels, examples)
    aupr = sk.average_precision_score(labels, examples)
    fpr = fpr_and_fdr_at_recall(labels, examples, recall_level)

    return auroc, aupr, fpr

import numpy as np

def compute_os_variance(os, th):
    """
    This function is borrowed from OWTTT (ICCV23): https://github.com/Yushu-Li/OWTTT
    Calculate the area of a rectangle.

    Parameters:
        os : OOD score queue.
        th : Given threshold to separate weak and strong OOD samples.

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

# for imagenet
def get_accuracy(args, score, labels, predicted, class_num=1000):

    threshold_range = np.arange(0, 1, 0.01)
    criterias = [compute_os_variance(np.array(score), th) for th in threshold_range]
    best_threshold = threshold_range[np.argmin(criterias)]
    
    unseen_mask = (score > best_threshold)
    
    predicted[unseen_mask] = class_num
    
    seen_labels = np.where(labels>=class_num, -1, labels)
    
    unseen_labels = np.where(labels>=class_num, class_num, -1)
    
    all_labels = np.where(labels>=class_num, class_num, labels)
    
    correct = (predicted == seen_labels) #id correct
    
    unseen_correct = (predicted == unseen_labels)
    
    all_correct = (predicted == labels)

    num_open = unseen_mask.sum() #number of ood samples

    id_err = np.sum((labels < class_num) & (predicted >= class_num))
    ood_err = np.sum((labels >= class_num) & (predicted < class_num))
    
    seen_acc = correct.sum() / max(len(correct) - num_open, 1)
    unseen_acc = unseen_correct.sum() / max(num_open, 1)
    h_score = 2 * seen_acc * unseen_acc / max(seen_acc + unseen_acc, 1e-8)
    
    print(f"Seen Acc: {seen_acc:.4f}, Unseen Acc: {unseen_acc:.4f}, H-score: {h_score:.4f}")
    
    #return seen_acc, unseen_acc, h_score, id_err, ood_err
    
    
def get_accuracy(args, score, labels, predicted, class_num=1000):

    threshold_range = np.arange(0, 1, 0.01)
    criterias = [compute_os_variance(np.array(score), th) for th in threshold_range]
    best_threshold = threshold_range[np.argmin(criterias)]
    
    unseen_mask = (score > best_threshold)
    
    predicted[unseen_mask] = class_num
    
    seen_labels = np.where(labels>=class_num, -1, labels)
    
    unseen_labels = np.where(labels>=class_num, class_num, -1)
    
    all_labels = np.where(labels>=class_num, class_num, labels)
    
    correct = (predicted == seen_labels) #id correct
    
    unseen_correct = (predicted == unseen_labels)
    
    all_correct = (predicted == labels)

    num_open = unseen_mask.sum() #number of ood samples

    id_err = np.sum((labels < class_num) & (predicted >= class_num))
    ood_err = np.sum((labels >= class_num) & (predicted < class_num))
    
    seen_acc = correct.sum() / max(len(correct) - num_open, 1)
    unseen_acc = unseen_correct.sum() / max(num_open, 1)
    h_score = 2 * seen_acc * unseen_acc / max(seen_acc + unseen_acc, 1e-8)
    
    print(f"Seen Acc: {seen_acc:.4f}, Unseen Acc: {unseen_acc:.4f}, H-score: {h_score:.4f}")
    

def get_accuracy_ood(args, unseen_mask, labels, predicted, class_num=1000):
    
    predicted[unseen_mask] = class_num
    
    seen_labels = np.where(labels>=class_num, -1, labels)
    
    unseen_labels = np.where(labels>=class_num, class_num, -1)
    
    all_labels = np.where(labels>=class_num, class_num, labels)
    
    correct = (predicted == seen_labels) #id correct
    
    unseen_correct = (predicted == unseen_labels)
    
    all_correct = (predicted == labels)

    num_open = unseen_mask.sum() #number of ood samples

    id_err = np.sum((labels < class_num) & (predicted >= class_num))
    ood_err = np.sum((labels >= class_num) & (predicted < class_num))
    
    seen_acc = correct.sum() / max(len(correct) - num_open, 1)
    unseen_acc = unseen_correct.sum() / max(num_open, 1)
    h_score = 2 * seen_acc * unseen_acc / max(seen_acc + unseen_acc, 1e-8)
    
    print(f"Seen Acc: {seen_acc:.4f}, Unseen Acc: {unseen_acc:.4f}, H-score: {h_score:.4f}")
    #args
def get_and_print_results(args, in_score, out_score, auroc_list, aupr_list, fpr_list):
    '''
    1) evaluate detection performance for a given OOD test set (loader)
    2) print results (FPR95, AUROC, AUPR)
    '''
    aurocs, auprs, fprs = [], [], []
    measures = get_measures(-in_score, -out_score)
    aurocs.append(measures[0]); auprs.append(measures[1]); fprs.append(measures[2])
    print(f'in score samples (random sampled): {in_score[:3]}, out score samples: {out_score[:3]}')

    auroc = np.mean(aurocs); aupr = np.mean(auprs); fpr = np.mean(fprs)
    auroc_list.append(auroc); aupr_list.append(aupr); fpr_list.append(fpr)  # used to calculate the avg over multiple OOD test sets
    print("FPR:{}, AUROC:{}, AURPC:{}".format(fpr, auroc, aupr))