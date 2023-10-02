#!/bin/bash

cnt=0

for file in configs/exps/*; do
    if [ -f "$file" ]; then
        echo "$file"
        
        ((cnt++))
        device=$(expr $cnt % 4) 
        echo Load_device $device 
        # export CUDA_VISIBLE_DEVICES=$device

        CUDA_VISIBLE_DEVICES=$device python main.py \
        --config $file \
        --device $device \
        --batch_size 48  &
        
        if [ $((cnt % 4)) -eq 0 ]; then
            echo "Number is divisible by 4, waiting..."
            wait 
            echo "Done waiting!"
        fi 
    fi
done
