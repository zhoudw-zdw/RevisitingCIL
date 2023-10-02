import json
import argparse
from trainer import train

def main():
    args = setup_parser().parse_args()
    tag = args.config.split('/')[-1].split('.json')[0]
    mode = args.mode
    param = load_json(args.config)
    args = vars(args)  # Converting argparse Namespace to a dict.
    args.update(param)  # Add parameters from json
    train(args, tag, mode)

def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)
    return param

def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorthms.')
    parser.add_argument('--config', type=str, default='./exps/finetune.json',
                        help='Json file of settings.')
    parser.add_argument('--mode', type=str, default='train')
    return parser

if __name__ == '__main__':
    main()