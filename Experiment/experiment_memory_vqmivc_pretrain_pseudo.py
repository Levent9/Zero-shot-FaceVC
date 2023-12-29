import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import random
from src.dataset import FaceSpeech_dataset
from models.vqmivc_encoder import Encoder, CPCLoss_sameSeq, Encoder_lf0
from models.vqmivc_decoder import Decoder_ac
from models import FACE_ENCODER
from models.vqmivc_mi_estimators import CLUBSample_group, CLUBSample_reshape
from src.logger import Logger
from torch.nn import DataParallel as DP
from torch.utils.data import DataLoader
from collections import OrderedDict
import numpy as np
from torch.autograd import grad
import copy
from glob import glob
from tqdm import tqdm
import soundfile as sf
from itertools import chain
from pathlib import Path
from src.scheduler import WarmupScheduler
import kaldiio
from Tools.preprocess.pwg_vqmivc_spectrogram import logmelspectrogram
import resampy
import pyworld as pw
import subprocess
import matplotlib.pyplot as plt
from Experiment.experiment_tools import extract_logmel, seed_worker


class ExperimentBuilder(nn.Module):
    def __init__(self,config):
        super(ExperimentBuilder, self).__init__()
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.enabled = True
        seed = config.getint("hparams","seed")
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

        self.is_train = config.getboolean("input", "is_train")
        self.if_provide_pseudo = config.getboolean("model", "if_provide_pseudo")
        self.train_if_decvtor = False
        if self.is_train:
            os.environ['CUDA_VISIBLE_DEVICES'] = config.get("hparams","gpu")
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = config.get("hparams","infer_gpu")

        self.output_path=config.get("output","output_dir")
        if config.getboolean("input","is_train"):
            self.logger = Logger(os.path.join(self.output_path, 'log'))
        self.config = config

        if config.get("model","encoder_lf0_type") == "no_emb":
            self.dim_lf0 = 1
        else:
            self.dim_lf0 = 64


        self.encoder = Encoder(in_channels=config.getint("model","encoder_in_channels"), 
                                channels= config.getint("model","encoder_channels"), 
                                    n_embeddings= config.getint("model","encoder_n_embeddings"), 
                                        z_dim = config.getint("model","encoder_z_dim"), 
                                            c_dim= config.getint("model","encodr_c_dim"))
        self.encoder_lf0 = Encoder_lf0(config.get("model","encoder_lf0_type"))
        self.cpc = CPCLoss_sameSeq(n_speakers_per_batch = config.getint("model","n_speakers_per_batch"),
                                    n_utterances_per_speaker = config.getint("model","n_utterances_per_speaker"),
                                    n_prediction_steps = config.getint("model","n_prediction_steps"),
                                    n_negatives = config.getint("model","n_negatives"),
                                    z_dim = config.getint("model","encoder_z_dim"), 
                                    c_dim= config.getint("model","encodr_c_dim"))

        self.encoder_spk = FACE_ENCODER[config.get("model","face_encoder")](config.getint("model", "slot_size"),config.getint("model", "slot_channel_size"))

        self.decoder = Decoder_ac(dim_neck=config.getint("model","encoder_z_dim"), dim_lf0=self.dim_lf0, use_l1_loss=True)
        self.speech_decoder = copy.deepcopy(self.decoder)

        self.encoder.cuda()
        self.encoder_lf0.cuda()
        self.cpc.cuda()
        self.encoder_spk.cuda()
        self.decoder.cuda()
        self.speech_decoder.cuda()

        self.all_models = [self.encoder, self.encoder_lf0, 
                                self.cpc, self.encoder_spk, self.decoder, self.speech_decoder]

        self.optimizer = torch.optim.Adam(
            chain(self.encoder_spk.parameters(), self.decoder.parameters()),
            lr=config.getfloat("hparams","scheduler_initial_lr"))   

        if config.getboolean("input","is_train"):
            root_path = Path(config.get("input","data_path"))
            self.train_data = FaceSpeech_dataset(
                root = root_path,
                n_sample_frames = config.getint("hparams","sample_frames"),
                mode='train' + '_' + config.get("input","spk_num"),
                face_type=config.get("model", "face_type"),
                speech_type=config.get("model", "speech_type"),
                if_provide_pseudo = self.if_provide_pseudo
            )

            self.valid_dataset = FaceSpeech_dataset(
                root = root_path,
                n_sample_frames = config.getint("hparams","sample_frames"),
                mode='valid' + '_' + config.get("input","spk_num"),
                face_type=config.get("model","face_type"),
                speech_type=config.get("model", "speech_type"),
                if_provide_pseudo = self.if_provide_pseudo
            )

            self.dataloader = DataLoader(
                self.train_data,
                batch_size=config.getint("hparams","batch_size"), # 256
                shuffle=True,
                num_workers=config.getint("hparams","n_works"),
                pin_memory=True,
                worker_init_fn=seed_worker,
                drop_last=False)

            self.valid_dataloader = DataLoader(
                self.valid_dataset,
                batch_size=config.getint("hparams","batch_size"), # 256
                shuffle=False,
                num_workers=config.getint("hparams","n_works"),
                pin_memory=True,
                worker_init_fn=seed_worker,
                drop_last=False)

            warmup_epochs = 2000 // (len(self.train_data) // config.getint("hparams","batch_size"))
            print('warmup_epochs:', warmup_epochs)
            self.scheduler = WarmupScheduler(
                self.optimizer,
                warmup_epochs = warmup_epochs,
                initial_lr = config.getfloat("hparams","scheduler_initial_lr"),
                max_lr = config.getfloat("hparams","scheduler_max_lr"),
                milestones = [config.getint("hparams","scheduler_milestones_0"), 
                                config.getint("hparams","scheduler_milestones_1"),
                                    config.getint("hparams","scheduler_milestones_2"),],
                gamma = config.getfloat("hparams","scheduler_gamma")
            )


        self.iteration = 0
        
        try:
            self.checkpont = config.get("model","checkpoint")
        except:
            self.checkpont = None

        if self.checkpont is not None:
            self.load_ckpt(os.path.join(self.output_path,'models',self.checkpont))
        else:
            self.start_epoch = 1
        
        self.load_pretrain(config.get("model", "pretrain_model_path"))
        
        self.kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
        self.lambda_spk_emb_recall = self.config.getfloat("hparams", "lambda_spk_emb_recall")
        self.lambda_address_recall = self.config.getfloat("hparams", "lambda_address_recall")
        self.lambda_speech_rec = self.config.getfloat("hparams", "lambda_speech_rec")
        self.lambda_diff_rec = self.config.getfloat("hparams", "lambda_diff_rec")

        try:
            self.if_address_mse = self.config.getboolean("model", "if_address_mse")
        except:
            self.if_address_mse = False
            
        try:
            self.if_decoder_no_grad = self.config.getboolean("model", "if_decoder_no_grad")
        except:
            self.if_decoder_no_grad = False

    def to_eval(self):
        for m in self.all_models:
            m.eval()

    def to_train(self):
        for m in self.all_models:
            m.train()

    def load_ckpt(self, ckpt_path):
        pass


    def load_pretrain(self, ckpt_path):
        pretrain_checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        if self.config.getboolean("model", "pretrain_encoder"):
            self.encoder.load_state_dict(pretrain_checkpoint['encoder'])
        if self.config.getboolean("model", "pretrain_decoder"):
            self.decoder.load_state_dict(pretrain_checkpoint['decoder'])
            self.speech_decoder.load_state_dict(pretrain_checkpoint['decoder'])
        if self.config.getboolean("model", "pretrain_cpc"):
            self.cpc.load_state_dict(pretrain_checkpoint['cpc'])
        if self.config.getboolean("model", "pretrain_speech_spk_encoder"):
            self.encoder_spk.speech_encoder.load_state_dict(pretrain_checkpoint['encoder_spk'])
        print(f'Load pretrain model from {ckpt_path}')


    def save_ckpt(self, epoch, ckpt_path=False):
        if not ckpt_path:
            # ckpt_path = os.path.join(self.output_root, self.output_path[2:], 'models', 'Best_model.rec_%.4f_ckpt-%04d.pt' % (val_mean[0], epoch))
            ckpt_path = os.path.join( self.output_path,'models','checkpoint', "model.ckpt-{:0>4d}.pt".format(epoch))                                   
            print(f"Saving Facevc model and optimizer at epoch {epoch} iteration {self.iteration} to {ckpt_path}")
        else:
            print(f"Saving Current Best Facevc model and optimizer at epoch {epoch} iteration {self.iteration} to {ckpt_path}")
        checkpoint_state = {
        "encoder": self.encoder.state_dict(),
        "encoder_lf0": self.encoder_lf0.state_dict(),
        "cpc": self.cpc.state_dict(),
        "encoder_spk": self.encoder_spk.state_dict(),
        "decoder": self.decoder.state_dict(),
        "optimizer": self.optimizer.state_dict(),
        "scheduler": self.scheduler.state_dict(),
        "epoch": epoch
        }
        torch.save(checkpoint_state, ckpt_path)
    
    def run_train_iter(self):
        raise NotImplementedError
    def run_valid_iter(self):
        raise NotImplementedError


    def run_valid(self):
        print('Please waiting! Valid')
        self.val_average_face_recon_loss = self.val_average_speech_recon_loss = self.val_average_speech_emb_recall_loss = self.val_average_address_recall_loss = 0
        self.to_eval()

        cur_epoch_metrics_val = []
        for self.val_idx, batch in enumerate(self.valid_dataloader, 1):
            metrics = self.run_valid_iter(batch)
            cur_epoch_metrics_val.append([v for k, v in metrics.items()])
        val_keys = [k for k in metrics.keys()]
        self.to_train()
    
        return cur_epoch_metrics_val, val_keys


    def run_experiment(self):
        best_score = 1000
        for epoch in range(self.start_epoch, self.config.getint("hparams","n_epochs")+1):
            self.average_face_recon_loss = self.average_speech_recon_loss = self.average_speech_emb_recall_loss = self.average_address_recall_loss = 0

            epoch_start_time = time.time()
            cur_epoch_metrics = []
            cur_epoch_metrics_val = []
            for self.idx, batch in enumerate(self.dataloader ,1):
                metrics = self.run_train_iter(batch)
                # self.logger.log_training(metrics, self.iteration)
                cur_epoch_metrics.append([v for k, v in metrics.items()])
                self.iteration = self.iteration + 1

            train_keys = [k for k in metrics.keys()]
            epoch_train_time = time.time() - epoch_start_time
            train_mean = np.mean(cur_epoch_metrics, axis=0)

            description = (', ').join(['{}: {:.4f}'.format(k, v) for k, v in zip(train_keys, train_mean)])
            Train_num = 'Epoch {:3d}  Inter {:4d}  '.format(epoch,self.iteration)
            description = Train_num + description
            print(description)

            if epoch == 1:
                cur_epoch_metrics_val, val_keys = self.run_valid()
                epoch_total_time = time.time() - epoch_start_time
                val_mean = np.mean(cur_epoch_metrics_val, axis=0)

            elif self.config.get("input","dataset") == 'lrs3' and epoch%5==0:
            # One epoch metrics
                cur_epoch_metrics_val, val_keys = self.run_valid()
                epoch_total_time = time.time() - epoch_start_time
                val_mean = np.mean(cur_epoch_metrics_val, axis=0)
                bests = glob(os.path.join( self.output_path, 'models', 'Best_*.pt'))
                bests.sort()
                if len(bests) > 3:
                    for prev in bests[3:]:
                        os.remove(prev)

                if val_mean[0] < best_score and epoch > 0.3 * (self.config.getint("hparams","n_epochs")+1):
                    ckpt_path = os.path.join( self.output_path, 'models', 'Best_model.rec_%.4f_ckpt-%04d.pt' % (val_mean[0], epoch))
                    best_score = val_mean[0]
                    self.save_ckpt(epoch, ckpt_path)

            elif self.config.get("input", "dataset") == 'vgg':
                cur_epoch_metrics_val, val_keys = self.run_valid()
                epoch_total_time = time.time() - epoch_start_time
                val_mean = np.mean(cur_epoch_metrics_val, axis=0)


            self.logger.log_epoch(train_mean, val_mean, train_keys,val_keys,
                epoch_train_time, epoch_total_time, epoch)
            self.scheduler.step()

            if not os.path.exists(os.path.join( self.output_path,'models','checkpoint')):
                os.makedirs(os.path.join( self.output_path,'models','checkpoint'))

            if (epoch % 50==0 ):
                self.save_ckpt(epoch)
                save_model_paths = glob(os.path.join( self.output_path, 'models', 'checkpoint','*.pt'))
                save_model_paths.sort()
                if len(save_model_paths) > 3:
                    for prev in save_model_paths[:3]:
                        os.remove(prev)
                


class Facevoice_memory_vqmivc_pretrain_pseudo(ExperimentBuilder):
    def __init__(self,config):
        super(Facevoice_memory_vqmivc_pretrain_pseudo,self).__init__(config)


    def mi_second_forward(self, mels, lf0, input_face, input_speech, speech_rec=False):
        self.optimizer.zero_grad()

        face, diff_face = input_face
        speech, diff_speech = input_speech        
        z, c, _, vq_loss, perplexity = self.encoder(mels)
        # print(mels[5,:])
        cpc_loss, accuracy = self.cpc(z, c)
        speech_emb, speech_emb_recall, face_emb, face_emb_recall, speech_address, face_address = self.encoder_spk(face, speech)
        diff_face_emb_recall= self.encoder_spk.forward_face(diff_face)
        diff_speech_emb = self.encoder_spk.forward_speech(diff_speech)
        # diff_face_emb_recall_1 = diff_face_emb_recall_1.detach()
        # diff_speech_emb_1 = diff_speech_emb_1.detach()        

        speech_emb_recall_loss = F.mse_loss(speech_emb, speech_emb_recall)
        address_recall_loss = self.kl_loss(face_address, speech_address)

        lf0_embs = self.encoder_lf0(lf0)
        face_recon_loss, pred_mels = self.decoder(z, lf0_embs, face_emb_recall, mels.transpose(1,2))
        face_pred_mels, face_pred_mels_postnet = self.decoder.forward_pseudo(z, lf0_embs, diff_face_emb_recall, mels.transpose(1,2), if_no_grad=self.if_decoder_no_grad)
        speech_pred_mels, speech_pred_mels_postnet = self.speech_decoder.forward_pseudo(z, lf0_embs, diff_speech_emb, mels.transpose(1,2), if_no_grad=self.if_decoder_no_grad)
        speech_recon_loss = torch.tensor(0.)

        diff_pred_recon_loss = torch.tensor(0.).cuda()


        diff_pred_recon_loss = F.mse_loss(face_pred_mels, speech_pred_mels_postnet) + \
                                F.l1_loss(face_pred_mels, speech_pred_mels_postnet) + \
                                F.l1_loss(face_pred_mels_postnet, speech_pred_mels_postnet) + \
                                F.mse_loss(face_pred_mels_postnet, speech_pred_mels_postnet)

        loss = face_recon_loss + self.lambda_spk_emb_recall * speech_emb_recall_loss + \
            self.lambda_address_recall * address_recall_loss + self.lambda_diff_rec * diff_pred_recon_loss
        
        loss.backward()

        if self.config.getboolean("model","train_if_clip"):
            torch.nn.utils.clip_grad_norm_(chain(self.encoder_spk.parameters(), self.decoder.parameters()), self.config.getfloat("hparams","clip_value"))

        self.optimizer.step()
        return face_recon_loss, speech_recon_loss, speech_emb_recall_loss, address_recall_loss, diff_pred_recon_loss


    def run_train_iter(self,batch):
        step_start_time = time.time()
        mels, lf0, speakers, face, speech, diff_face, diff_speech,= batch

        mels = mels.cuda() 
        lf0 = lf0.cuda()
        # print(speakers)
        face = face.cuda().squeeze(1) # (256, 1, 512)
        diff_face= diff_face.cuda().squeeze(1)
        input_face = [face, diff_face]

        speech = speech.cuda().squeeze(1)
        diff_speech= diff_speech.cuda().squeeze(1)
        input_speech = [speech, diff_speech]

        face_recon_loss, speech_recon_loss, speech_emb_recall_loss, address_recall_loss, diff_pred_recon_loss = self.mi_second_forward(mels, lf0, input_face, input_speech)
        
        metrics = OrderedDict()
        metrics['face_recon_loss'] = face_recon_loss
        metrics['speech_recon_loss'] = speech_recon_loss
        metrics['diff_pred_recon_loss'] = diff_pred_recon_loss
        metrics['speech_emb_recall_loss'] = speech_emb_recall_loss
        metrics['address_recall_loss'] = address_recall_loss
        metrics['Steptime'] = time.time()-step_start_time
        metrics['lr'] = self.optimizer.param_groups[0]['lr']
        return metrics


    def run_valid_iter(self,batch):

        mels, lf0, speakers, face, speech, diff_face, diff_speech= batch
        mels = mels.cuda() 
        lf0 = lf0.cuda()

        face = face.cuda().squeeze(1) # (256, 1, 512)
        diff_face= diff_face.cuda().squeeze(1)
        input_face = [face, diff_face]

        speech = speech.cuda().squeeze(1)
        diff_speech= diff_speech.cuda().squeeze(1)
        input_speech = [speech, diff_speech]

        with torch.no_grad():
            z, c, z_beforeVQ, vq_loss, perplexity = self.encoder(mels)
            speech_emb, speech_emb_recall, face_emb, face_emb_recall, speech_address_log, face_address_log = self.encoder_spk(face, speech)
            diff_face_emb_recall = self.encoder_spk.forward_face(diff_face)
            diff_speech_emb = self.encoder_spk.forward_speech(diff_speech)
            
            speech_emb_recall_loss = F.mse_loss(speech_emb, speech_emb_recall)
            address_recall_loss = self.kl_loss(face_address_log, speech_address_log)

            lf0_embs = self.encoder_lf0(lf0)
            face_recon_loss, pred_mels = self.decoder(z, lf0_embs, face_emb_recall, mels.transpose(1,2))
            face_pred_mels, face_pred_mels_postnet = self.decoder.forward_pseudo(z, lf0_embs, diff_face_emb_recall, mels.transpose(1,2))
            speech_pred_mels, speech_pred_mels_postnet = self.speech_decoder.forward_pseudo(z, lf0_embs, diff_speech_emb, mels.transpose(1,2))
            diff_pred_recon_loss = torch.tensor(0.).cuda()

            diff_pred_recon_loss = F.mse_loss(face_pred_mels, speech_pred_mels_postnet) + \
                                        F.l1_loss(face_pred_mels, speech_pred_mels_postnet) + \
                                        F.l1_loss(face_pred_mels_postnet, speech_pred_mels_postnet) + \
                                        F.mse_loss(face_pred_mels_postnet, speech_pred_mels_postnet)


        metrics = OrderedDict()
        metrics['val_face_recon_loss'] = face_recon_loss
        metrics['val_speech_emb_recall_loss'] = speech_emb_recall_loss
        metrics['val_address_recall_loss'] = address_recall_loss
        metrics['val_diff_pred_recon_loss'] = diff_pred_recon_loss
        return metrics


    def get_src_tar_paths(self, infer_dataset='LRS3'):
        src_speaker_dict = {}
        tar_speaker_dict = {}
        
        src_speaker_cont_path =  'test_src_speakers.txt'
        tar_speaker_cont_path = 'test_tar_speakers.txt'

        src_speaker_cont_f = open(src_speaker_cont_path,'r')
        src_speaker_cont = src_speaker_cont_f.readlines()
        
        for line in src_speaker_cont:
            spk_id = line.split('-')[0]
            gender = line.split('-')[1][:-1]
            src_speaker_dict[spk_id] = gender

        tar_speaker_cont_f = open(tar_speaker_cont_path,'r')
        tar_speaker_cont = tar_speaker_cont_f.readlines()
        for line in tar_speaker_cont:
            spk_id = line.split('-')[0]
            gender = line.split('-')[-1][:-1]
            tar_speaker_dict[spk_id] = gender

        select_src_wav_paths = []
        select_tar_wav_paths = []
        for i in src_speaker_dict:
            cur_spk_paths = glob(os.path.join(self.config.get("input", "wav_path"),i,'*.wav'))
            random.shuffle(cur_spk_paths)
            for wav_path in cur_spk_paths[:6]:
                select_src_wav_paths.append(wav_path)

        for i in tar_speaker_dict:      
            cur_spk_paths = glob(os.path.join(self.config.get("input", "wav_path"),i,'*.wav'))
            random.shuffle(cur_spk_paths)
            for wav_path in cur_spk_paths[:3]:
                select_tar_wav_paths.append(wav_path)
        print(len(select_src_wav_paths), len(select_tar_wav_paths))
        return src_speaker_dict, tar_speaker_dict, select_src_wav_paths, select_tar_wav_paths
        
        
    def get_save_path(self, src_wav_path, ref_wav_path):
        src_spk = src_wav_path.split('/')[-2]
        ref_spk = ref_wav_path.split('/')[-2]

        src_wav_id = src_wav_path.split('/')[-1][:-4]
        ref_wav_id = ref_wav_path.split('/')[-1][:-4]
        output_filename_suffix =  ref_spk[:4] + '_' + ref_wav_id + '_'+ \
            src_spk[:4] + '_' + src_wav_id + '_' + self.src_speaker_dict[src_spk] + '2' + self.tar_speaker_dict[ref_spk]
        return output_filename_suffix
        

    def run_inference(self, infer_dataset="LRS3"):
        self.infer_dataset = infer_dataset
        
        self.src_speaker_dict, self.tar_speaker_dict, select_src_wav_paths, select_tar_wav_paths = self.get_src_tar_paths(infer_dataset=self.infer_dataset)
        checkpoint_model_name = self.config.get("output", "checkpoint")
        if 'Best' in checkpoint_model_name:
            checkpoint_path = os.path.join(self.output_root, self.output_path[2:], 'models', checkpoint_model_name)
        else:
            checkpoint_path = os.path.join(self.output_root, self.output_path[2:], 'models', 'checkpoint', checkpoint_model_name)
        
        output_dir = os.path.join(self.output_root, self.output_path[2:],'wav_'+checkpoint_model_name.split('-')[-1][:-3])
        print('Load From:')
        print(checkpoint_path)
        os.makedirs(output_dir, exist_ok=True)
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.encoder_spk.load_state_dict(checkpoint['encoder_spk'])
        self.decoder.load_state_dict(checkpoint['decoder'])

        self.encoder.eval()
        self.encoder_spk.eval()
        self.decoder.eval()

        mel_stats = np.load(os.path.join(self.config.get("input","data_path"),"mel_stats_" + self.config.get("input","spk_num") + ".npy"))

        mean = mel_stats[0]
        std = mel_stats[1]


        feat_writer = kaldiio.WriteHelper("ark,scp:{o}.ark,{o}.scp".format(o=str(output_dir)+'/feats.1'))
        for src_wav_path in tqdm(select_src_wav_paths):
            for ref_wav_path in select_tar_wav_paths:
                mel, lf0 = extract_logmel(src_wav_path, mean, std)
                ref_mel, _ = extract_logmel(ref_wav_path, mean, std)
                ref_speaker_id = ref_wav_path.split('/')[-2]
                ref_wav_id = ref_wav_path.split('/')[-1][:-4]
                
                face_emb = np.load(os.path.join(self.config.get("input","data_path"),'test',self.config.get("model","face_type").split('_')[0],
                                ref_speaker_id, ref_speaker_id + '_' + ref_wav_id +'.npy'))
                mel = torch.FloatTensor(mel.T).unsqueeze(0).cuda()
                lf0 = torch.FloatTensor(lf0).unsqueeze(0).cuda()
                ref_mel = torch.FloatTensor(ref_mel.T).unsqueeze(0).cuda()
                face_emb = torch.FloatTensor(face_emb).cuda()

                out_filename = os.path.join(self.get_save_path(src_wav_path, ref_wav_path))
                with torch.no_grad():
                    
                    z, _, _, _ = self.encoder.encode(mel)
                    lf0_embs = self.encoder_lf0(lf0)
                    spk_embs = self.encoder_spk.inference(face_emb)
                    output = self.decoder(z, lf0_embs, spk_embs)
                    
                    logmel = output.squeeze(0).cpu().numpy()

                    feat_writer[out_filename] = logmel

        feat_writer.close()
        

        print('synthesize waveform...')

        # replace the pwg checkpoint save path 
        cmd = ['parallel-wavegan-decode', '--checkpoint', \
            '/home/zysheng/US_Facevc/pretrained/vqmivc/VQMIVC-pretrained models/vocoder/checkpoint-3000000steps.pkl', \
            '--feats-scp', f'{str(output_dir)}/feats.1.scp', '--outdir', str(output_dir)]
        subprocess.call(cmd)
        


            



            


            

            


