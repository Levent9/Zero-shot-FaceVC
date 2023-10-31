import numpy as np
import torch
from torch.utils.data import Dataset
import json
import random
from pathlib import Path
import os
import time
import copy
import pandas as pd
from glob import glob
import sys


class FaceSpeech_dataset(Dataset):
    def __init__(self, root, n_sample_frames, mode, face_type, speech_type, dataset="lrs3", 
                    if_provide_pseudo = False):
        self.root = Path(root)
        self.n_sample_frames = n_sample_frames
        self.if_provide_pseudo = if_provide_pseudo

        self.speakers = sorted(os.listdir(str(root)+'/train/mels_' +mode.split('_')[1]))
        self.mode2file = {'train':'pretrain', 'valid':'trainval'}
        
        with open(self.root / f"{mode}.json") as file:
            metadata = json.load(file)
        self.metadata = []

        self.dataset = dataset

        for mel_len, mel_out_path, lf0_out_path in metadata:
            face_path = mel_out_path.replace("/mels_" + mode.split('_')[1] + "/","/" + face_type.split('_')[0] + "/")
            if not os.path.exists(face_path):
                continue
            speech_emb_path = mel_out_path.replace("/mels_" + mode.split('_')[1] + "/","/" + speech_type + "/")
            
            mel_out_path = Path(mel_out_path)
            lf0_out_path = Path(lf0_out_path)
            speaker = mel_out_path.parent.stem
            self.metadata.append([speaker, str(mel_out_path), str(lf0_out_path), face_path, speech_emb_path])
        print('mode:', mode, 'n_sample_frames:', n_sample_frames, 'metadata:', len(self.metadata))
        random.shuffle(self.metadata)

        self.face_type = face_type
    


    def get_random_frames_padding(self, face):
        frame_length = int(self.face_type.split('_')[-1])
        cur_length = face.shape[0]
        if cur_length <= frame_length:
            if len(face.shape)==3:
                face_return = np.zeros((frame_length, 1, 512), dtype=np.float32)
            else:
                face_return = np.zeros((frame_length, 3, 128, 128), dtype=np.float32)
            face_return[:cur_length] = face
        else:
            start_index = random.choice(range(face.shape[0] - frame_length))
            face_return = face[start_index: start_index + frame_length]
        return face_return
    

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):

        speaker, mel_path, lf0_path, face_path, speech_emb_path = self.metadata[index]
    
        # mel_path = self.root.parent / mel_path
        # lf0_path = self.root.parent / lf0_path
        # face_path = self.root.parent / face_path
        # speech_emb_path = self.root.parent / speech_emb_path
        time_ = time.time()
        mel = np.load(mel_path).T
        lf0 = np.load(lf0_path)
        face = np.load(face_path)
        if self.face_type == 'facesemb':
            len_list = list(range(face.shape[0]))
            face = face[random.choice(len_list)]

        speech_emb = np.load(speech_emb_path)

        melt = mel
        lf0t = lf0
        while mel.shape[-1] < self.n_sample_frames:
            mel = np.concatenate([mel, melt], -1)
            lf0 = np.concatenate([lf0, lf0t], 0)
        zero_idxs = np.where(lf0 == 0.0)[0]
        nonzero_idxs = np.where(lf0 != 0.0)[0]
        if len(nonzero_idxs) > 0 :
            mean = np.mean(lf0[nonzero_idxs])
            std = np.std(lf0[nonzero_idxs])
            if std == 0:
                lf0 -= mean
                lf0[zero_idxs] = 0.0
            else:
                lf0 = (lf0 - mean) / (std + 1e-8)
                lf0[zero_idxs] = 0.0

        pos = random.randint(0, mel.shape[-1] - self.n_sample_frames)
        mel = mel[:, pos:pos + self.n_sample_frames] 
        lf0 = lf0[pos:pos + self.n_sample_frames] 

        if self.if_provide_pseudo:
            diff_spk_idx = random.choice(list(range(len(self.metadata))))
            diff_speaker, diff_mel_path, diff_lf0_path, diff_face_path, diff_speech_emb_path = self.metadata[diff_spk_idx]
            while diff_speaker == speaker:
                diff_spk_idx = random.choice(list(range(len(self.metadata))))
                diff_speaker, diff_mel_path, diff_lf0_path, diff_face_path, diff_speech_emb_path = self.metadata[diff_spk_idx]

            # diff_face_path = self.root.parent / diff_face_path
            # diff_speech_emb_path = self.root.parent / diff_speech_emb_path 
            diff_face = np.load(diff_face_path)
            diff_speech_emb = np.load(diff_speech_emb_path)
           # print(mel.shape, lf0.shape, face.shape, speech_emb.shape, diff_face.shape, diff_speech_emb.shape)
            return torch.from_numpy(mel), torch.from_numpy(lf0), self.speakers.index(speaker), \
                    torch.from_numpy(face), torch.from_numpy(speech_emb), torch.from_numpy(diff_face), \
                    torch.from_numpy(diff_speech_emb)

        return torch.from_numpy(mel), torch.from_numpy(lf0), self.speakers.index(speaker), \
        torch.from_numpy(face), torch.from_numpy(speech_emb) # os.path.basename(face_path) # self.speakers.index(speaker),
