#!/bin/bash

cnt=0
export CUDA_VISIBLE_DEVICES=1,2,3

for file in configs/exps/*; do
    if [[ $file =~ .*adam_vt_deep_loss_cross_entropy.* ]]; then
        echo "$file"
        
        # ((cnt++))
        # device=$(expr $cnt % 4) 
        # echo Load_device $device 
        # export CUDA_VISIBLE_DEVICES=$device

        python main.py \
        --config $file \
        --mode train \
        --tuned_epoch 10  # &
        
        # if [ $((cnt % 4)) -eq 0 ]; then
        #     echo "Number is divisible by 4, waiting..."
        #     wait 
        #     echo "Done waiting!"
        # fi 
    fi
done
