#!/bin/bash
# custom config
TRAINER=sct

DATA=$1
DATASET=$2
CFG=$3  # config file
CTP=$4  # class token position (end or middle)
NCTX=$5  # number of context tokens
SHOTS=$6  # number of shots (1, 2, 4, 8, 16)
CSC=$7  # class-specific context (False or True)
lambda=$8 # 0.3 0.25 0.2; do
topk=$9 
current_time=$(date "+%Y-%m-%d-%H-%M-%S")

K=5
LAM=0.001
BS=128

for arg in "$@"; do
    case $arg in
        --K=*)   K="${arg#*=}"; shift;;
        --LAM=*) LAM="${arg#*=}"; shift;;
        --BS=*) BS="${arg#*=}"; shift;;        
    esac
done

for SEED in 1; do
  for lambda in ${lambda}; do
    DIR=output/${TRAINER}_${DATASET}/${CFG}_${SHOTS}shots_lambda_${lambda}_nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
    if [ -d "$DIR" ]; then
        echo "Oops! The results exist at ${DIR} (so skip this job)"
    else
        #echo $PWD
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/LoCoOp/${CFG}.yaml \
        --output-dir ${DIR} \
        --lambda_value ${lambda} \
        --topk ${topk} \
        TRAINER.sct.N_CTX ${NCTX} \
        TRAINER.sct.CSC ${CSC} \
        TRAINER.sct.CLASS_TOKEN_POSITION ${CTP} \
        DATASET.NUM_SHOTS ${SHOTS}
    fi
    bash scripts/locoop/eval_tta.sh ${DATA} imagenet ${CFG} 1 ${DIR} ${SEED} ${K} ${LAM} ${BS}
  done
done