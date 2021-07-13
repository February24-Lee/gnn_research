#!/bin/bash

python ./src/train_new.py \
    --debug \
    --datapath data// \
    --seed 42 \
    --dataset citeseer \
    --type mutigcn \
    --nhiddenlayer 4 \
    --nbaseblocklayer 0 \
    --hidden 128 \
    --epoch 400 \
    --lr 0.009 \
    --weight_decay 0.003 \
    --early_stopping 400 \
    --sampling_percent 1 \
    --dropout 0.3 \
    --normalization BingGeNormAd \
    --task_type semi \
    --withloop \
    --withbn \
    --wandb_name citeseer_gcn_nodrop \
    
