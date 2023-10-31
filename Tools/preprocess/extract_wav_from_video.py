import numpy as np
from glob import glob
import os
import subprocess


if __name__ == "__main__":
    # Change to your lrs3 dataset root
    dataset_root = "Dataset/LRS3"
    f = open("/home/zysheng/Github/Zero-shot-FaceVC/Tools/lrs3_speakers.txt", 'r')
    spk_ids = [i.split(":")[0] for i in f.readlines()[:200]]
    
    for type_ in ["pretrain", "trainval"]:
        for spk_id in spk_ids:
            wav_root = os.path.join(dataset_root, type_ + "_wav_200", spk_id)
            os.makedirs(wav_root, exist_ok=True)
            video_paths = glob(os.path.join(dataset_root, type_, spk_id, "*.mp4"))
            for video_path in video_paths:
                save_path = os.path.join(wav_root, video_path.split("/")[-1].replace(".mp4",".wav"))
                command = "ffmpeg -i {} -vn -acodec pcm_s16le -ar 16000 -ac 1 {}".format(video_path, save_path)
                res = subprocess.call(command, shell=True)