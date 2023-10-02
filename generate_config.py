import os
import json

# 1. Read the template file in json format
with open('./configs/minghaocil_omnibenchmark.json', 'r') as f:
    config = json.load(f)

# 2. Set the parameters
for opt in ['sgd','adam']:
    for lr in [0.002,0.005,0.008,0.01,0.02,0.03,0.04,0.05]:
        for wd in [0.0005,0.001,0.002,0.003,0.004,0.005]:
            if opt == 'adam':
                continue
            with open(f'./configs/exps2/minghaocil_lr({lr})_wd({wd})_opt({opt}).json', 'w') as f:
                config['weight_decay'] = wd
                config['init_lr'] = lr
                config['optimizer'] = opt
                config["device"] = ["0","1","2","3"]
                config["batch_size"] = int(768/4*len(config["device"]))
                config["tuned_epoch"] = 20
                json.dump(config, f, indent=4)