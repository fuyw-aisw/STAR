#!/bin/bash

CUDA_VISIBLE_DEVICES=0 bash scripts/locoop/train.sh {data_path} imagenet vit_b16_ep50 end 16 1 False 0.25 200
wait

CUDA_VISIBLE_DEVICES=0 bash scripts/sct/train.sh {data_path} imagenet vit_b16_ep25 end 16 1 False 0.4 200
wait