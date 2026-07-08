import argparse
import torch
from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer
import numpy as np
from utils.train_eval_util import set_val_loader, set_ood_loader_ImageNet, set_id_ood_loader
from utils.detection_util import get_and_print_results, get_accuracy, get_accuracy_ood
from utils.plot_util import plot_distribution
import trainers.locoop
import datasets.imagenet


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.lambda_value:
        cfg.lambda_value = args.lambda_value

    if args.topk:
        cfg.topk = args.topk


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.TRAINER.LOCOOP = CN()
    cfg.TRAINER.LOCOOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.LOCOOP.CSC = False  # class-specific context
    cfg.TRAINER.LOCOOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.LOCOOP.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.LOCOOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new
    cfg.INFERENCE = CN() 
    cfg.INFERENCE.THRESHOLD_TYPE = "adaptive"
    cfg.INFERENCE.LAM = 0.1
    cfg.INFERENCE.ALPHA = 0.1
    cfg.INFERENCE.IS_CONF = True
    cfg.INFERENCE.Q = 0.05
    cfg.INFERENCE.K = 5
    cfg.INFERENCE.ENTROPY_QUEUE_LENGTH = 64
    cfg.INFERENCE.TAU_SELECTION_TYPE = "adaptive"
    cfg.INFERENCE.META_RATIO = 0.2
    cfg.INFERENCE.MAX_SIZE_PER_CLUSTER = 64
    cfg.INFERENCE.CLASS_NUM = 1000
    cfg.INFERENCE.THRESHOLD = 60  
    cfg.INFERENCE.METHOD = "CLIP"
    cfg.INFERENCE.OPTIM = CN()
    cfg.INFERENCE.OPTIM.NAME = "SGD"
    cfg.INFERENCE.OPTIM.LR = 0.0001
    cfg.INFERENCE.OPTIM.LR_SCHEDULER = "cosine"
    cfg.INFERENCE.OPTIM.MOMENTUM = 0.9
    cfg.INFERENCE.OPTIM.WEIGHT_DECAY = 0  
    
    
def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    
    if args.inference_config:
        cfg.merge_from_file(args.inference_config)        

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg


def main(args):
    import clip_w_local
    cfg = setup_cfg(args)
    _, preprocess = clip_w_local.load(cfg.MODEL.BACKBONE.NAME)

    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))

    if args.in_dataset in ['imagenet']:
        out_datasets = ['Texture']#'iNaturalist',

        
    if not args.inference_config:
        
        trainer = build_trainer(cfg)

        trainer.load_model(args.model_dir, epoch=args.load_epoch)

        id_data_loader = set_val_loader(args, preprocess)
        print('*********\n*********')
        print(f'Evaluating on ID dataset:')
        in_score_mcm, in_score_gl, id_accuracy, id_labels, id_predicted, in_score = trainer.test_ood(id_data_loader, args.T, id_flag=True)
        print(f'Classification of ID dataset:{(id_accuracy*100):.1f}%')
        print('*********\n*********')

        auroc_list_mcm, aupr_list_mcm, fpr_list_mcm = [], [], []
        auroc_list_gl, aupr_list_gl, fpr_list_gl = [], [], []


        for out_dataset in out_datasets:
            print(f"Evaluting OOD dataset {out_dataset}")
            ood_loader = set_ood_loader_ImageNet(args, out_dataset, preprocess)
            out_score_mcm, out_score_gl, ood_labels, ood_predicted, ood_score = trainer.test_ood(ood_loader, args.T, id_flag=False)

            print("MCM score")
            get_and_print_results(args, in_score_mcm, out_score_mcm,
                                  auroc_list_mcm, aupr_list_mcm, fpr_list_mcm)

            print('******')

            print("GL-MCM score")
            get_and_print_results(args, in_score_gl, out_score_gl,
                                  auroc_list_gl, aupr_list_gl, fpr_list_gl)
            get_accuracy(args, np.concatenate([in_score, ood_score]), np.concatenate([id_labels, ood_labels]), np.concatenate([id_predicted, ood_predicted]))                               

            print('*********\n*********')

            plot_distribution(args, in_score_mcm, out_score_mcm, out_dataset, score='MCM')
            plot_distribution(args, in_score_gl, out_score_gl, out_dataset, score='GLMCM')

        print("MCM avg. FPR:{}, AUROC:{}, AUPR:{}".format(np.mean(fpr_list_mcm), np.mean(auroc_list_mcm), np.mean(aupr_list_mcm)))
        print("GL-MCM avg. FPR:{}, AUROC:{}, AUPR:{}".format(np.mean(fpr_list_gl), np.mean(auroc_list_gl), np.mean(aupr_list_gl)))


        return
    else:
        auroc_list, aupr_list, fpr_list = [], [], []
        trainer = build_trainer(cfg)

        if cfg.INFERENCE.METHOD == "CLIP":
            trainer.load_clip_model()
        else:
            trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.init_tta()
        for out_dataset in out_datasets:
            print(f"Evaluting OOD dataset {out_dataset}")
            
            args.class_num = cfg.INFERENCE.CLASS_NUM
            data_loader = set_id_ood_loader(args, args.in_dataset, out_dataset, preprocess)

            is_ood_all, max_conf_all, labels_all, predicted_all = trainer.test_time_adapt(data_loader)

            is_ood_all = np.array(is_ood_all)
            max_conf_all = np.array(max_conf_all)
            labels_all = np.array(labels_all)
            predicted_all = np.array(predicted_all)
            valid_mask = (labels_all >= 0)
            is_ood_all = is_ood_all[valid_mask]
            max_conf_all = max_conf_all[valid_mask]
            labels_all = labels_all[valid_mask]
            predicted_all = predicted_all[valid_mask]

            mask = labels_all == cfg.INFERENCE.CLASS_NUM
            get_and_print_results(args, -max_conf_all[~mask], -max_conf_all[mask], auroc_list, aupr_list, fpr_list)
            get_accuracy_ood(args, is_ood_all, labels_all, predicted_all)

            #breakpoint()
            
        print("ours avg. FPR:{}, AUROC:{}, AUPR:{}".format(np.mean(fpr_list), np.mean(auroc_list), np.mean(aupr_list)))
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument('--in_dataset', default='imagenet', type=str,
                        choices=['imagenet'], help='in-distribution dataset')
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--inference-config", type=str, default="", help="path to inference config file"
    )    
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    # augment for LoCoOp
    parser.add_argument('--lambda_value', type=float, default=1,
                        help='temperature parameter')
    parser.add_argument('--topk', type=int, default=200,
                        help='topk')
    # augment for MCM and GL-MCM
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        help='mini-batch size')
    parser.add_argument('--T', type=float, default=1,
                        help='temperature parameter')
 
    args = parser.parse_args()
    main(args)
