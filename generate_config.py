import os
import json

# 1. Read the template file in json format
with open('./configs/minghaocil_omnibenchmark.json', 'r') as f:
    config = json.load(f)

# 2. Set the parameters
for loss_fn in ['cross_entropy','focal_loss']:
    for vpt_type in ['deep','shallow']:
        for opt in ['sgd','adam']:
            for lr in [0.005,0.01,0.02,0.03,0.04,0.05]:
                for wd in [0.0005,0.001,0.002,0.003,0.004,0.005]:
                    # if opt == 'adam':
                    #     continue
                    # if vpt_type == 'shallow':
                    #     continue
                    # if loss_fn == 'cross_entropy':
                    #     continue
                    with open(f'./configs/exps/minghaocil_lr_{lr}_wd_{wd}_opt_{opt}_vt_{vpt_type}_loss_{loss_fn}.json', 'w') as f:
                        config['weight_decay'] = wd
                        config['init_lr'] = lr
                        config['optimizer'] = opt
                        config["device"] = ["0","1","2","3"]
                        config["vpt_type"] = vpt_type
                        config["batch_size"] = int(768/4*len(config["device"]))
                        config["tuned_epoch"] = 20
                        config["loss_fn"] = loss_fn
                        config["prompt_token_num"] = 5
                        json.dump(config, f, indent=4)