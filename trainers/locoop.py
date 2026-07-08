import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip_w_local import clip
from clip_w_local.simple_tokenizer import SimpleTokenizer as _Tokenizer
import numpy as np
from tqdm import tqdm
from PIL import Image
from trainers.retrieval import compute_fisher_score, AdaptiveMemoryBankTTA
from trainers.retrieval import *
from utils.detection_util import get_and_print_results

_tokenizer = _Tokenizer()
softmax = nn.Softmax(dim=1).cuda()


def entropy_select_topk(p, top_k, label, num_of_local_feature):
    """
    Extract non-Top-K regions and calculate entropy.
    """
    label_repeat = label.repeat_interleave(num_of_local_feature)
    p = F.softmax(p, dim=-1)
    pred_topk = torch.topk(p, k=top_k, dim=1)[1]
    contains_label = pred_topk.eq(torch.tensor(label_repeat).unsqueeze(1)).any(dim=1)
    selected_p = p[~contains_label]

    if selected_p.shape[0] == 0:
        return torch.tensor([0]).cuda()
    return -torch.mean(torch.sum(selected_p * torch.log(selected_p+1e-5), 1))


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x, _, _, _ = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.LOCOOP.N_CTX
        ctx_init = cfg.TRAINER.LOCOOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.LOCOOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.LOCOOP.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image, return_embeds=False):
        image_features, local_image_features = self.image_encoder(image.type(self.dtype))

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        local_image_features = local_image_features / local_image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()

        logits = logit_scale * image_features @ text_features.t()
        logits_local = logit_scale * local_image_features @ text_features.T
        if not return_embeds:
            return logits, logits_local
        else:
            return logits, image_features

    def get_text_features(self):
        text_features = []
        prompts = self.prompt_learner() # cls x 77 x 512
        tokenized_prompts = self.tokenized_prompts # cls x 77
        t_features = self.text_encoder(prompts, tokenized_prompts)
        text_features.append(t_features / t_features.norm(dim=-1, keepdim=True)) # 10x512
        text_features = torch.stack(text_features, dim=0) # 1x10x512

        return torch.mean(text_features, dim=0)
    
@TRAINER_REGISTRY.register()
class LoCoOp(TrainerX):
    """Local regularized Context Optimization (LoCoOp).
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.LOCOOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        self.lambda_value = cfg.lambda_value
        self.top_k = cfg.topk

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.LOCOOP.PREC == "fp32" or cfg.TRAINER.LOCOOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.LOCOOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)


    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        prec = self.cfg.TRAINER.LOCOOP.PREC

        if prec == "amp":
            with autocast():
                output, output_local = self.model(image)
                # calculate CoOp loss
                loss_id = F.cross_entropy(output, label)

                # calculate OOD regularization loss
                batch_size, num_of_local_feature = output_local.shape[0], output_local.shape[1]
                output_local = output_local.view(batch_size * num_of_local_feature, -1)
                loss_en = - entropy_select_topk(output_local, self.top_k, label, num_of_local_feature)

                # calculate total loss for LoCoOp
                loss = loss_id + self.lambda_value * loss_en

            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output, output_local = self.model(image)

            # calculate CoOp loss
            loss_id = F.cross_entropy(output, label) 

            # calculate OOD regularization loss
            batch_size, num_of_local_feature = output_local.shape[0], output_local.shape[1]
            output_local = output_local.view(batch_size * num_of_local_feature, -1)     
            loss_en = - entropy_select_topk(output_local, self.top_k, label, num_of_local_feature)

            # calculate total loss for LoCoOp
            loss = loss_id + self.lambda_value * loss_en

            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "loss_id": loss_id.item(),
            "loss_en": loss_en.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)

    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            output = self.model_inference(input)
            if len(output) == 2:
                output = output[0]
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]

    @torch.no_grad()
    def test_ood(self, data_loader, T, id_flag):
        """Test-time OOD detection pipeline."""
        to_np = lambda x: x.data.cpu().numpy()
        concat = lambda x: np.concatenate(x, axis=0)

        self.set_model_mode("eval")
        self.evaluator.reset()

        glmcm_score = []
        mcm_score = []
        correct = 0
        labels_all = []
        predicted_all = []
        #predicted_global = []
        ood_score_all = []
        #result_mcm, result_gl = {"fpr":[], "auroc":[]}, {"fpr":[], "auroc":[]}
        for batch_idx, (images, labels) in enumerate(tqdm(data_loader)):
            images = images.cuda()
            output, output_local = self.model_inference(images)
            
            
            if id_flag:
                correct += (output.argmax(1) == labels.cuda()).float().cpu().sum().item()
                labels_all.extend(labels.cpu().numpy())
            else:
                labels_all.extend(output.shape[0] * [output.shape[1]])
            
            predicted_all.extend(output.argmax(1).cpu().numpy())
            
            score = to_np(F.softmax(output / T, dim=-1))
            ood_score = 1-np.max(score, axis=1)
            ood_score_all.extend(ood_score)
            
            output /= 100.0
            output_local /= 100.0
            smax_global = to_np(F.softmax(output / T, dim=-1))
            smax_local = to_np(F.softmax(output_local / T, dim=-1))
            #print(T, smax_global.min(), smax_global.max())
            mcm_global_score = -np.max(smax_global, axis=1)
            mcm_local_score = -np.max(smax_local, axis=(1, 2))
            mcm_score.extend(mcm_global_score)
            glmcm_score.extend(mcm_global_score + mcm_local_score)
        '''

            mask = np.array(labels_all) == 1000
            
            print(np.array(labels_all).shape, np.array(mcm_score).shape)
            fpr, auroc = get_and_print_results(np.array(mcm_score).squeeze()[~mask], np.array(mcm_score).squeeze()[mask], [], [], [])
            #print(fpr, auroc)
            fpr_gl, auroc_gl = get_and_print_results(np.array(glmcm_score).squeeze()[~mask], np.array(glmcm_score).squeeze()[mask], [], [], [])
            result_mcm['fpr'].append(fpr)
            result_mcm['auroc'].append(auroc)
            result_gl['fpr'].append(fpr_gl)
            result_gl['auroc'].append(auroc_gl) 
            
        return result_mcm, result_gl
        '''
        if id_flag:
            return concat(mcm_score)[:len(data_loader.dataset)].copy(), concat(glmcm_score)[:len(data_loader.dataset)].copy(), correct / len(data_loader.dataset), labels_all, predicted_all, ood_score_all
        else:
            return concat(mcm_score)[:len(data_loader.dataset)].copy(), concat(glmcm_score)[:len(data_loader.dataset)].copy(), labels_all, predicted_all, ood_score_all


    @torch.no_grad()
    def test_visualize(self, img_path, label):
        """code for visualization results"""
        self.set_model_mode("eval")
        self.evaluator.reset()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/16", device=device)

        image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
        output, output_local = self.model_inference(image)

        num_regions = output_local.shape[1]
        label = torch.tensor(label).cuda()
        label_repeat = label.repeat_interleave(num_regions)
        output_local = F.softmax(output_local, dim=-1)

        output_local = output_local.view(num_regions, -1)

        # -----top 200--------
        pred_topk = torch.topk(output_local, k=200, dim=1)[1]
        contains_label = pred_topk.eq(torch.tensor(label_repeat).unsqueeze(1)).any(dim=1)

        return contains_label



    def init_tta(self):
        self.model_ = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        self.entropy_queue = deque(maxlen=self.cfg.INFERENCE.ENTROPY_QUEUE_LENGTH)

        if self.cfg.INFERENCE.CLASS_NUM <= 10:
            self.use_meta_cluster = False
        else:
            self.use_meta_cluster= True

        if not self.use_meta_cluster:
            self.cluster_num = self.cfg.INFERENCE.CLASS_NUM 
        else:
            self.cluster_num = max(int(self.cfg.INFERENCE.META_RATIO * self.cfg.INFERENCE.CLASS_NUM), 10)
        self.cluster_num = min(self.cluster_num, self.cfg.INFERENCE.CLASS_NUM)

        if self.cfg.MODEL.BACKBONE.NAME == "ViT-L/14":
            feat_dim = 768
        elif self.cfg.MODEL.BACKBONE.NAME in ["ViT-B/16", "ViT-B/32"]:
            feat_dim = 512
        elif self.cfg.MODEL.BACKBONE.NAME == "RN50":
            feat_dim = 1024
        else:
            print("Input hidden dimension")
        self.device = next(self.model_.image_encoder.parameters()).device
        '''
        if isinstance(self.model, nn.DataParallel):
            self.device = next(self.model.module.image_encoder.parameters()).device 
        else:
            self.device = next(self.model.image_encoder.parameters()).device
        '''

        self.memory_threshold = 0
        self.memory_bank = AdaptiveMemoryBankTTA(
            class_num=self.cfg.INFERENCE.CLASS_NUM,
            embed_dim=feat_dim,
            cluster_num=self.cluster_num,
            max_size_per_cluster=self.cfg.INFERENCE.MAX_SIZE_PER_CLUSTER,
            alpha=self.cfg.INFERENCE.ALPHA,
            warmup_N=5,
            device=self.device
        )
        self.ctx_params = [p for name, p in self.model_.prompt_learner.named_parameters() if name == "ctx"]        
        #self.tta_optim = build_optimizer(self.ctx_params, self.cfg.INFERENCE.OPTIM)
        #self.tta_sched = build_lr_scheduler(self.tta_optim, self.cfg.INFERENCE.OPTIM)

        from torch.optim import SGD
        from torch.optim.lr_scheduler import CosineAnnealingLR
        self.tta_optim = SGD(
            self.ctx_params,
            lr=self.cfg.INFERENCE.OPTIM.LR,
            momentum=self.cfg.INFERENCE.OPTIM.MOMENTUM,
            weight_decay=self.cfg.INFERENCE.OPTIM.WEIGHT_DECAY)
        
        self.tta_sched = CosineAnnealingLR(self.tta_optim, T_max=50, eta_min=1e-6)
        
        self.register_model("tta_ctx", self.model_.prompt_learner, self.tta_optim, self.tta_sched)   
        self.tta_scaler = GradScaler() if self.cfg.TRAINER.LOCOOP.PREC == "amp" else None
        
        
    def _set_adapt_mode(self):
        self.set_model_mode("eval")
        self.evaluator.reset()
        for p in self.model_.parameters():
            p.requires_grad = False

        for name, p in self.model_.prompt_learner.named_parameters():
            p.requires_grad = (name == "ctx")

        self.model_.prompt_learner.train()       

    def test_time_adapt(self, data_loader):
        self._set_adapt_mode()
        is_ood_all, max_conf_all, labels_all, predicted_all = [], [], [], []
        if self.use_meta_cluster:
            with torch.no_grad():
                text_features = self.model_.get_text_features()
                self.memory_bank.assign_clusters(text_features.float())
            
        #result = {"fpr":[], "auroc":[]}      
        for batch_idx, (images, labels) in enumerate(tqdm(data_loader)):
            self.model_.train()
            images = images.cuda()
            #with torch.no_grad():
            
            logits, image_features = self.model_(images, return_embeds=True)
            logits, image_features = logits.float(), image_features.float()
            labels_all.extend(labels.cpu().numpy())

            predicted_all.extend(logits.argmax(1).cpu().numpy())

            p = F.softmax(logits, dim=1)
            entropy = -(p * torch.log(p + 1e-12)).sum(dim=1)
            self.entropy_queue.extend(entropy.detach().cpu().tolist())
            #print(self.entropy_queue)

            # adaptive tau
            if self.cfg.INFERENCE.TAU_SELECTION_TYPE == "adaptive":
                threshold_range = np.arange(0, np.log(self.cfg.INFERENCE.CLASS_NUM), 0.01)
                criterias = [compute_fisher_score(np.array(self.entropy_queue), th) for th in threshold_range]
                tau = threshold_range[np.argmax(criterias)]
            else:
                tau = 0.6 * np.log(self.cfg.INFERENCE.CLASS_NUM)

            ceu_loss = CEULoss(logits, self.cfg.INFERENCE.K, tau, self.cfg.INFERENCE.LAM)
            #ceu_loss = softmax_entropy(logits).mean(0)

            self.tta_optim.zero_grad()
            #print(tau.item(), ceu_loss.item())
            ceu_loss.backward()
            self.tta_optim.step()

            
            #print(self.memory_threshold)    
            with torch.no_grad():
                max_conf = F.softmax(logits.float()/100, dim=1).max(dim=1)[0]
                max_conf_all.extend(max_conf.cpu().numpy())
                if self.cfg.INFERENCE.THRESHOLD_TYPE == "adaptive":
                    self.memory_threshold = 0.01 * self.memory_threshold + 0.99 * np.percentile(np.array(max_conf_all), self.cfg.INFERENCE.THRESHOLD)
                self.memory_bank.update_example_bank(logits.float(), image_features, self.memory_threshold)
                #print(self.memory_bank.example_bank['embed'].shape[0])
                self.memory_bank.update_prototypes(is_conf=self.cfg.INFERENCE.IS_CONF)
                self.memory_bank.rank_thresholds(q=self.cfg.INFERENCE.Q)

                is_ood = self.memory_bank.detect_ood(logits.float(), image_features)
                is_ood_all.extend(is_ood.cpu().numpy())

        return is_ood_all, max_conf_all, labels_all, predicted_all


    def load_clip_model(self):
        #self.device = torch.device("cuda:0")
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        self.lambda_value = cfg.lambda_value
        self.top_k = cfg.topk

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.LOCOOP.PREC == "fp32" or cfg.TRAINER.LOCOOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)