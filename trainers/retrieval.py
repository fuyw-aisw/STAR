import torch
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
from collections import deque

### softmax

def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

# update prototype

def prototype_update(example_bank, prototype=None, cls_prototype=None, alpha=0.3, cls_to_cluster=None):
    if example_bank is None:
        return
    
    embeds = example_bank["embed"]
    pseudo_labels = example_bank["pseudo_label"]
    logits = example_bank["logit"]
    probs = torch.softmax(logits, dim=1)
    conf, _ = torch.max(probs, dim=1)
    class_num = logits.shape[1]
    #_, topk_idx = torch.topk(probs, k=top_k, dim=1)
    hidden_dim = embeds.shape[1]
    if cls_to_cluster is None:
        #cluster_var = torch.zeros(class_num, hidden_dim, device=embeds.device)
        if prototype is not None:
            prototype_new = prototype.clone()
        else:
            prototype_new = torch.zeros(class_num, hidden_dim, device=embeds.device)
        exist_cls = torch.unique(pseudo_labels)
        for cls in exist_cls:
            mask = (pseudo_labels == cls)
            weights = conf[mask]
            if mask.sum() > 0:
                mean_emb = torch.sum(weights.unsqueeze(1) * embeds[mask], dim=0) / (weights.sum() + 1e-12)
                if prototype is None:
                    prototype_new[cls] = mean_emb
                else:                
                    prototype_new[cls] = alpha * mean_emb + (1-alpha) * prototype[cls]
        #return prototype_new
                #cluster_var[cls] = torch.sum(weights.unsqueeze(1) * (embeds[mask] - mean_emb)**2, dim=0) / (weights.sum() + 1e-12)
                
    else:
        cluster_num = len(torch.unique(cls_to_cluster))
        #cluster_var = torch.zeros(cluster_num, hidden_dim, device=embeds.device)
        if prototype is not None:
            prototype_new = prototype.clone()
        else:
            prototype_new = torch.zeros(cluster_num, hidden_dim, device=embeds.device)
        '''
        if cls_prototype is not None:
            cls_prototype_new = cls_prototype.clone()
        else:
            cls_prototype_new = torch.zeros(class_num, hidden_dim, device=embeds.device)
            
        exist_cls = torch.unique(pseudo_labels)
        for cls in exist_cls:
            mask = (pseudo_labels == cls)
            if mask.sum() > 0:
                weights = conf[mask]
                mean_emb = torch.sum(weights.unsqueeze(1) * embeds[mask], dim=0) / (weights.sum() + 1e-12)
                if cls_prototype is None:
                    cls_prototype_new[cls] = mean_emb
                else:
                    cls_prototype_new[cls] = alpha * mean_emb + (1 - alpha) * cls_prototype[cls]
        '''            
        cluster_pseudo_labels = cls_to_cluster[pseudo_labels]
        exist_clu = torch.unique(cluster_pseudo_labels)
        for clu in exist_clu:
            mask = (cluster_pseudo_labels == clu)
            weights = conf[mask]
            if mask.sum() > 0:
                mean_emb = torch.sum(weights.unsqueeze(1) * embeds[mask], dim=0) / (weights.sum() + 1e-12)
                if prototype is None:
                    prototype_new[clu] = mean_emb
                else:                
                    prototype_new[clu] = alpha * mean_emb + (1-alpha) * prototype[clu]
                #cluster_var[cls] = torch.sum(weights.unsqueeze(1) * (embeds[mask] - mean_emb)**2, dim=0) / (weights.sum() + 1e-12)
                               
    return prototype_new#, cls_prototype_new



## example bank update
def example_bank_update(embeds, logits, example_bank=None, threshold=0.6, max_size_per_cluster=64, cls_to_cluster=None):
    """
    Update the example bank with new samples efficiently.
    """
    probs = torch.softmax(logits, dim=1)
    max_probs, pseudo_labels = torch.max(probs, dim=1)
    mask_add = (max_probs >= threshold)
    if mask_add.sum() == 0:
        return example_bank

    embeds_add = embeds[mask_add]
    logits_add = logits[mask_add]
    pseudo_labels_add = pseudo_labels[mask_add]

    # initialize or append
    if example_bank is None:
        example_bank = {
            "embed": embeds_add,
            "logit": logits_add,
            "pseudo_label": pseudo_labels_add
        }
    else:
        example_bank["embed"] = torch.cat([example_bank["embed"], embeds_add], dim=0)
        example_bank["logit"] = torch.cat([example_bank["logit"], logits_add], dim=0)
        example_bank["pseudo_label"] = torch.cat([example_bank["pseudo_label"], pseudo_labels_add], dim=0)

    # efficient truncation by label
    pseudo_labels_all = example_bank["pseudo_label"]
    _, sorted_idx = torch.sort(pseudo_labels_all)  # group by label
    sorted_labels = pseudo_labels_all[sorted_idx]

    # find boundaries for each label
    unique_labels, counts = torch.unique_consecutive(sorted_labels, return_counts=True)
    cumsum = torch.cumsum(counts, dim=0)
    keep_indices = []

    start = 0
    for i, (label, count) in enumerate(zip(unique_labels, counts)):
        end = cumsum[i]
        label_indices = sorted_idx[start:end]
        # keep last `max_size_per_cluster` if too many
        if count > max_size_per_cluster:
            label_indices = label_indices[-max_size_per_cluster:]
        keep_indices.append(label_indices)
        start = end

    keep_indices = torch.cat(keep_indices)

    # apply filtering
    example_bank["embed"] = example_bank["embed"][keep_indices]
    example_bank["logit"] = example_bank["logit"][keep_indices]
    example_bank["pseudo_label"] = example_bank["pseudo_label"][keep_indices]
    if cls_to_cluster is not None:
        example_bank["cluster_label"] = cls_to_cluster[example_bank["pseudo_label"]]
    return example_bank


def rank_update(example_bank, prototype, cls_prototype=None, alpha=0.1, beta=1, cls_to_cluster=None, top_k=5):
    embeds = example_bank["embed"]
    pseudo_labels = example_bank["pseudo_label"]
    logits = example_bank["logit"]
    
    probs = torch.softmax(logits, dim=1)
    conf, _ = torch.max(probs, dim=1)
    #_, topk_idx = torch.topk(probs, k=top_k, dim=1) 
    #class_num = prototype.shape[0]
    class_num = logits.shape[1]
    N = embeds.shape[0]    
    exist_cls = torch.unique(pseudo_labels)
    embeds_norm = F.normalize(embeds, dim=1)
    proto_norm = F.normalize(prototype, dim=1)
    
    if cls_to_cluster is None:
        #sims = torch.zeros(N, device=embeds.device)
        rank_thresholds = torch.zeros(class_num, device=embeds.device)
        #sim_means = torch.zeros(class_num, device=embeds.device)
        #sim_stds = torch.zeros(class_num, device=embeds.device)
        #ssims = []
        for cls in range(class_num):
            mask = (pseudo_labels == cls)
            #if mask.sum() == 0:
            #mask = (topk_idx == cls).any(dim=1)
            mask_neg = ~mask
            neg_indices = torch.nonzero(mask_neg).squeeze()

            if mask.sum() > 0:
                emb_cls = embeds_norm[mask]
                proto_cls = proto_norm[cls].unsqueeze(0)
                sim_cls = torch.matmul(emb_cls, proto_cls.T).squeeze(1)
                #print(sim_cls.mean().item(), sim_cls.std().item())
                rank_thresholds[cls] = torch.quantile(sim_cls.float(), alpha)
                #print(rank_thresholds[cls], torch.quantile(sim_cls.float(), 0.1)) 
            else:
                rank_thresholds[cls] = torch.mean(rank_thresholds[:cls])
        return rank_thresholds
            
    else:
      
        cluster_labels = cls_to_cluster[example_bank["pseudo_label"]]
        cluster_num = len(torch.unique(cls_to_cluster))
        rank_thresholds = torch.zeros(cluster_num, device=embeds.device)
        rank_thresholds_cls = torch.zeros(class_num, device=embeds.device)
        #cls_proto_norm = F.normalize(cls_prototype, dim=1)
        for clu in range(cluster_num):
            mask = (cluster_labels == clu)
            if mask.sum() > 0:
                emb_clu = embeds_norm[mask]
                proto_clu = proto_norm[clu].unsqueeze(0)
                sim_clu = torch.matmul(emb_clu, proto_clu.T).squeeze(1)
                #if cluster_var is not None:
                #    sim_cls = sim_cls / (1 + cluster_var[cls].mean())

                sim_neighbor = torch.matmul(emb_clu, emb_clu.T)
                sim_neighbor.fill_diagonal_(0)
                k = min(top_k, emb_clu.shape[0]-1)
                if k > 0:
                    topk_sims, _ = torch.topk(sim_neighbor, k, dim=1)
                    s_bar = topk_sims.mean(dim=1)
                else:
                    s_bar = torch.zeros_like(sim_clu)
                #beta = 1
                sim_correct = sim_clu / (1 + beta * (1 - s_bar))

                #lam = torch.sigmoid(sim_cls-s_bar)
                               
                #sim_correct = (sim_cls + s_bar) / 2
                #beta = 1
                
                #print(sim_correct.mean().item(), sim_correct.std().item())
                #sim_correct = sim_cls * torch.pow(conf[mask], 0.5)
                rank_thresholds[clu] = torch.quantile(sim_correct.float(), alpha)
                '''
                if cls_prototype is not None:
                    cls_in_cluster = torch.unique(pseudo_labels[mask])
                    if len(cls_in_cluster) > 0:
                        for cls_id in cls_in_cluster:
                            mask_cls = (pseudo_labels == cls_id)
                            emb_cls = embeds_norm[mask_cls]
                            proto_cls = cls_proto_norm[cls_id].unsqueeze(0)
                            sim_cls = torch.matmul(emb_cls, proto_cls.T).squeeze(1)
                            threshold = torch.quantile(sim_cls.float(), alpha)
                            rank_thresholds_cls[cls_id] = 0.7 * threshold + 0.3 * rank_thresholds[clu]
                '''
                            
                            
            else:
                 rank_thresholds[clu] = torch.mean(rank_thresholds[:clu]) if clu > 0 else 0.0
        return rank_thresholds#, rank_thresholds_cls    


class AdaptiveMemoryBankTTA:
    def __init__(self, class_num, embed_dim, cluster_num,
                 max_size_per_cluster=64, alpha=0.2, 
                 warmup_N=10, device='cuda:0'):
        self.class_num = class_num
        self.embed_dim = embed_dim
        self.cluster_num = cluster_num
        self.max_size_per_cluster = max_size_per_cluster
        self.alpha = alpha
        self.warmup_N = warmup_N
        self.device = device

        self.example_bank = None  # CPU tensors
        self.prototypes = torch.zeros(class_num, embed_dim, device=device)
        self.cluster_prototypes = torch.zeros(cluster_num, embed_dim, device=device)
        self.class_thr = torch.zeros(class_num, device=device)
        self.cluster_thr = torch.zeros(cluster_num, device=device)
        self.global_thr = 0.1
        self.num_seen_per_class = torch.zeros(class_num, dtype=torch.int)
        self.cls_to_cluster = None

    def assign_clusters(self, text_features):
        text_features_norm = F.normalize(text_features, dim=1)
        cls_to_cluster = KMeans(n_clusters=self.cluster_num, random_state=42, n_init=20).fit_predict(text_features_norm.detach().cpu().numpy())
        
        self.prototypes = F.normalize(text_features, dim=1).to(self.device)
        
        if isinstance(cls_to_cluster, np.ndarray):
            self.cls_to_cluster = torch.from_numpy(cls_to_cluster).to(self.device)

    def update_prototypes(self, is_conf=False):
        """
        Update per-class and per-cluster prototypes using EMA
        """
        if self.example_bank is None:
            return
        embeds = self.example_bank['embed'].to(self.device)
        labels = self.example_bank['pseudo_label'].to(self.device)
        logits = self.example_bank["logit"].to(self.device)
        #probs = torch.softmax(logits, dim=1)
        #conf, _ = torch.max(probs, dim=1)
        conf = self.example_bank["conf"].to(self.device)
        embeds_norm = F.normalize(embeds, dim=1)

        # per-class prototype
        for c in range(self.class_num):
            mask = (labels == c)
            if mask.sum() == 0: 
                continue
            if is_conf:
                emb_mean = torch.sum(conf[mask].unsqueeze(1) * embeds_norm[mask], dim=0) / (conf[mask].sum() + 1e-12)
            else:
                emb_mean = F.normalize(embeds_norm[mask].mean(dim=0), dim=0)
                
            self.prototypes[c] = F.normalize((1-self.alpha)*self.prototypes[c] + self.alpha*emb_mean, dim=0)

        # cluster-level prototype
        #print(self.cls_to_cluster)
        if self.cls_to_cluster is not None:
            for clu in range(self.cluster_num):
                #print(clu)
                
                cls_in_clu = (self.cls_to_cluster == clu).nonzero(as_tuple=True)[0]
                #print(cls_in_clu)
                #breakpoint()
                if cls_in_clu.numel() == 0:
                    continue
                mask = torch.zeros(labels.shape[0], dtype=torch.bool, device=self.device)
                for c in cls_in_clu:
                    mask |= (labels == c)
                if mask.sum() == 0:
                    continue
                if is_conf:
                    emb_mean = torch.sum(conf[mask].unsqueeze(1) * embeds_norm[mask], dim=0) / (conf[mask].sum() + 1e-12)
                else:
                    emb_mean = F.normalize(embeds_norm[mask].mean(dim=0), dim=0)
                    
                #emb_mean = F.normalize(embeds_norm[mask].mean(dim=0), dim=0)
                self.cluster_prototypes[clu] = F.normalize((1-self.alpha)*self.cluster_prototypes[clu] + self.alpha*emb_mean, dim=0)

    def rank_thresholds(self, q=0.1):
        """
        class margin;
        cluster margin;
        global margin
        """
        if self.example_bank is None:
            return
        embeds = self.example_bank['embed'].to(self.device)
        labels = self.example_bank['pseudo_label'].to(self.device)
        logits = self.example_bank["logit"].to(self.device)
        probs = torch.softmax(logits, dim=1)
        conf, _ = torch.max(probs, dim=1)
        
        embeds_norm = F.normalize(embeds, dim=1)
        proto_norm = F.normalize(self.prototypes, dim=1)
        sims = embeds_norm @ proto_norm.T
        energy = torch.logsumexp(logits, dim=1)

        #id_score = 0.5 * sims[torch.arange(len(labels)), labels] + 0.5 * (1 - energy)
        
        id_score = sims[torch.arange(len(labels)), labels] 
        
        for c in range(self.class_num):
            mask = (labels == c)
            if mask.sum() < self.warmup_N:
                # fallback to cluster/global margin
                if self.cls_to_cluster is not None:
                    clu = self.cls_to_cluster[c]
                    self.class_thr[c] = self.cluster_thr[clu]
                else:
                    self.class_thr[c] = self.global_thr
            else:
                vals = id_score[mask].float()
                self.class_thr[c] = torch.quantile(vals, q).item()
        #print(self.margin_thr)
        # update cluster margin using text similarity
        if self.cls_to_cluster is not None:
            for clu in range(self.cluster_num):
                cls_in_clu = (self.cls_to_cluster == clu).nonzero(as_tuple=True)[0]
                #print(cls_in_clu)
                #breakpoint()
                mask = torch.zeros(labels.shape[0], dtype=torch.bool, device=labels.device)
                for c in cls_in_clu:
                    mask |= (labels == c)
                if mask.sum() > 0:
                    vals = id_score[mask].float()
                    self.cluster_thr[clu] = torch.quantile(vals, q).item()

        self.global_thr = id_score.quantile(q).item()


    def detect_ood(self, logits, embeds):
        """
        logits: [B, C], embeds: [B, D]
        returns: is_ood: [B], diagnostics: dict
        """
        probs = F.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        embed_norm = F.normalize(embeds, dim=1)
        proto_norm = F.normalize(self.prototypes, dim=1)
        sims = embed_norm @ proto_norm.T
        
        sim = sims[torch.arange(len(pred)), pred]
        energy = torch.logsumexp(logits, dim=1)
        #id_score = 0.5 * sim + 0.5 * (1-energy)
        id_score = sim
        thr = self.class_thr[pred]
        is_ood = id_score < thr

        return is_ood

    def update_example_bank(self, logits, embeds, memory_threshold=0.3):
        """
        Add only safe ID samples to memory bank
        """
        probs = F.softmax(logits, dim=1)
        confs, preds = probs.max(dim=1)

        is_id_mask = (confs >= memory_threshold)

        embeds_add = embeds[is_id_mask].detach().cpu()
        logits_add = logits[is_id_mask].detach().cpu()
        pseudo_labels_add = preds[is_id_mask].detach().cpu()
        confs_add = confs[is_id_mask].detach().cpu()

        if self.example_bank is None:
            self.example_bank = {'embed': embeds_add, 'logit': logits_add, 'pseudo_label': pseudo_labels_add, 'conf': confs_add}
        else:
            self.example_bank['embed'] = torch.cat([self.example_bank['embed'], embeds_add], dim=0)
            self.example_bank['logit'] = torch.cat([self.example_bank['logit'], logits_add], dim=0)
            self.example_bank['pseudo_label'] = torch.cat([self.example_bank['pseudo_label'], pseudo_labels_add], dim=0)
            self.example_bank['conf'] = torch.cat([self.example_bank['conf'], confs_add], dim=0)

        # enforce per-class max size FIFO
        labels_all = self.example_bank['pseudo_label']
        embeds_all = self.example_bank['embed']
        logits_all = self.example_bank['logit']
        confs_all = self.example_bank['conf']
        keep_idx = []
        unique_labels = torch.unique(labels_all)
        for cls in unique_labels:
            idxs = (labels_all == cls).nonzero(as_tuple=True)[0]
            if idxs.numel() > self.max_size_per_cluster:
                idxs = idxs[-self.max_size_per_cluster:]
            keep_idx.append(idxs)
        keep_idx = torch.cat(keep_idx)
        self.example_bank['embed'] = embeds_all[keep_idx]
        self.example_bank['logit'] = logits_all[keep_idx]
        self.example_bank['pseudo_label'] = labels_all[keep_idx]
        self.example_bank['conf'] = confs_all[keep_idx]

        #return diag

def alpha_rank_threshold(sim_pos, sim_neg, alpha):
    ind_pos = (sim_pos > alpha)
    sum_pos = ind_pos.sum()

    ind_neg = (sim_neg < alpha)
    sum_neg = ind_neg.sum()
    
    if sum_pos == 0 or sum_neg == 0:
        return float('inf')
    
    mean_pos = torch.sum(sim_pos * ind_pos) / sum_pos
    mean_neg = torch.sum(sim_neg * ind_neg) / sum_neg
    
    intra_var = torch.sum((sim_pos - mean_pos) ** 2) / sum_pos + torch.sum((sim_neg - mean_neg) ** 2) / sum_neg 
    inter_gap = (mean_pos - mean_neg) ** 2
    result = intra_var - inter_gap
    if isinstance(result, torch.Tensor):
        return result.item()
    return result #intra_var - inter_gap
#tau

def tau_from_entropy(logits):
    p = F.softmax(logits, dim=1)
    log_p = torch.log(p + 1e-12)
    entropy = - (p * log_p).sum(dim=1)
    tau = torch.quantile(entropy, 0.9)
    return tau
    
def compute_fisher_score(entropy, tau):
    id_set = entropy[entropy < tau]
    ood_set = entropy[entropy > tau]
    if len(id_set) < 2 or len(ood_set) < 2:
        return 0
    w_id, w_ood = len(id_set) / len(entropy), len(ood_set) / len(entropy)
    mu_id, mu_ood = id_set.mean(), ood_set.mean()
    var_id, var_ood = id_set.var(ddof=0), ood_set.var(ddof=0)
    between = w_id * w_ood * (mu_id - mu_ood)**2
    within = w_id * var_id + w_ood * var_ood
    fisher = between / (within + 1e-12)
    return fisher


def prototype_cl_loss(embeds, logits, prototype, cl_tau=0.1):
    N = logits.shape[0]
    probs = F.softmax(logits, dim=-1)
    _, pseudo_labels = torch.max(probs, dim=1)
    embeds = F.normalize(embeds, dim=1)
    prototype = F.normalize(prototype, dim=1)
    sim = torch.matmul(embeds, prototype.T) / cl_tau
    numerator = torch.exp(sim[torch.arange(N), pseudo_labels])
    denominator = torch.exp(sim).sum(dim=1)
    loss = -torch.log(numerator / denominator)
    loss = loss.mean()
    return loss
