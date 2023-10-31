from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
import numpy as np
from PIL import Image
import os
from glob import glob
from joblib import Parallel, delayed
from tqdm import tqdm
import torch


def extract_face_embs(video_path, mtcnn, resnet, spk_id, wav_id, save_root):
    face_imgs = []
    face_embs = []
    capture = cv2.VideoCapture(video_path)
    i = 0
    if capture.isOpened():
        while True:
            ret, img = capture.read()
            try:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_cropped = mtcnn(img)
                # cv2.imwrite(os.path.join('visual_pic/1',str(i)+'.jpg'), img_cropped.detach().cpu().numpy().transpose(1,2,0)*255)
                img_embedding = resnet(img_cropped.unsqueeze(0))
                face_imgs.append(img_cropped.detach().cpu().numpy())
                face_embs.append(img_embedding.detach().cpu().numpy())
            except:
                pass
            if not ret:
                break
            i = i + 1

    try:
        face_embs = np.stack(face_embs, axis=0)
        face_imgs = np.stack(face_imgs, axis=0)
        face_embs_mean = np.mean(face_embs, axis=0)
        os.makedirs(os.path.join(save_root,'facesemb',spk_id), exist_ok = True)
        np.save(os.path.join(save_root,'facesemb',spk_id, spk_id+'_'+wav_id+'.npy'), face_embs)
        os.makedirs(os.path.join(save_root,'facesimg',spk_id), exist_ok = True)
        np.save(os.path.join(save_root,'facesimg',spk_id, spk_id+'_'+wav_id+'.npy'), face_imgs)
        os.makedirs(os.path.join(save_root,'facesembmean',spk_id), exist_ok = True)
        np.save(os.path.join(save_root,'facesembmean',spk_id, spk_id+'_'+wav_id+'.npy'), face_embs_mean)
        
    except:
        print(f"Error: {spk_id}, {wav_id}")


if __name__ == "__main__":
    """
    # demo
    mtcnn = MTCNN(image_size=128, margin=50)
    resnet = InceptionResnetV1(pretrained='vggface2').eval()

    video_path = '/home/zysheng/mnt_215_disk1/Dataset/LRS3/test/81Ub0SMxZQo/00002.mp4'
    spk_id = '81Ub0SMxZQo'
    wav_id = '00002'
    save_root = '/home/zysheng/mnt_215_disk2/Dataset/LRS3/pwg_vqmivc/train'
    extract_face_embs(video_path, mtcnn, resnet, spk_id, wav_id, save_root)
    """
    
    mtcnn = MTCNN(image_size=128, margin=50)
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    
    dataset_root = "Dataset/LRS3"
    trans_dict = {"pretrain_wav_200":"train", "trainval_wav_200":"valid"}
    for type_ in ["pretrain_wav_200", "trainval_wav_200"]:
        video_paths = [i.replace(".wav", ".mp4").replace("_wav_200","") for i in glob(os.path.join(dataset_root, type_, "*/*.wav"))]
        spk_ids = [i.split("/")[-2] for i in video_paths]
        wav_ids = [os.path.basename(i)[:-4] for i in video_paths]
        save_root = "Dataset/LRS3/pwg_vqmivc/" + trans_dict[type_]
        Parallel(n_jobs=4)(delayed(extract_face_embs)(video_paths[i], mtcnn, resnet, spk_ids[i], wav_ids[i], save_root) for i in tqdm(range(len(video_paths))))