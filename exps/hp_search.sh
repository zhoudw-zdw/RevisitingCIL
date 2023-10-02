#!/bin/bash

for file in configs/exps/*; do
    if [ -f "$file" ]; then
        echo "$file"
        
        python main.py --config $file
    fi
done
