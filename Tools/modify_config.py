from configparser import ConfigParser
import argparse
from glob import glob
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file',required=True)
    parser.add_argument('--inference_gpu',required=True, type=str)
    parser.add_argument('--output_root', required=True)
    args = parser.parse_args()
    return args
    

if __name__ == '__main__':
    args = parse_args()
    conf_path = args.config_file
    config = ConfigParser()
    config.read(conf_path)
    print('------------------------------')
    print(conf_path)

    try:
        checkpoint = config.get("output", "checkpoint")
    except:
        print(' Change config file done ')
        ckpt_names = glob(os.path.join(args.output_root, config.get("output","output_dir")[2:],"models/checkpoint/*.pt"))
        ckpt_epoches = [int(os.path.basename(i).split("-")[1][:-3]) for i in ckpt_names]
        ckpt_epoch = max(ckpt_epoches)
        config.set("input","is_train","False")
        config.set("output","checkpoint","model.ckpt-{}.pt".format(ckpt_epoch))
        config.set("hparams","infer_gpu", args.inference_gpu)
        with open(conf_path,'w') as configfile:
            config.write(configfile)