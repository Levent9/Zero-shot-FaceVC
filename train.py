from Experiment import EXPERIMENTS
from configparser import ConfigParser
from shutil import copyfile
import warnings
import argparse
import os



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file',required=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    config = ConfigParser()
    config.read(os.path.join('./configs', args.config_file))
    
    os.makedirs(config.get("output","output_dir"),exist_ok=True)
    copyfile(os.path.join('./configs',args.config_file), os.path.join(config.get("output","output_dir"),"config.ini"))

    cur_exp = EXPERIMENTS[config.get("model","exp_type")](config)
    cur_exp.run_experiment()