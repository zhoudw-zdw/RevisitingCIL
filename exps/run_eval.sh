#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3

python main.py \
    --config configs/exps2/minghaocil_lr_0.03_wd_0.001_opt_sgd_vt_deep_loss_cross_entropy.json \
    --mode train-eval 