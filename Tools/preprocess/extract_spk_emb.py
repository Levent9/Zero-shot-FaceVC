from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
import numpy as np
import os
from glob import glob
import warnings
from joblib import Parallel, delayed
from tqdm import tqdm
warnings.filterwarnings("ignore")

lrs3_dict = {'pretrain':'train', 'trainval': 'valid'}


def extract_res_emb(wav_path, save_path, encoder):
    fpath = Path(wav_path)
    wav = preprocess_wav(fpath) 
    embed = encoder.embed_utterance(wav)
    os.makedirs(save_path.replace(os.path.basename(save_path),''),exist_ok=True)
    np.save(save_path, embed)
    
    
def extract_lrs3_dataset(mode):
    wav_paths = glob(os.path.join("Dataset/LRS3",mode+"_wav_200", "*/*.wav"))
    save_paths = [i.replace(mode+"_wav_200", "pwg_vqmivc/"+lrs3_dict[mode] +"/rese+emb") for i in wav_paths]
    encoder = VoiceEncoder()
    Parallel(n_jobs=6)(delayed(extract_res_emb)(wav_paths[i], save_paths[i], encoder) for i in tqdm(range(len(wav_paths))))
    

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    for mode in ['pretrain','trainval']:
        extract_lrs3_dataset(mode)