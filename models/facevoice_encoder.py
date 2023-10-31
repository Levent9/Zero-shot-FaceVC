# Menory network Encoder

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from models.speech_encoder import speech_encoder_att
from models.face_encoder import face_encoder_att
from models.basic_layers import Repara
import torch.nn.init as init


class facevoice_memorynet(nn.Module):
    def __init__(self, slot_size, slot_channel_size):
        super(facevoice_memorynet, self).__init__()
        self.speech_encoder = speech_encoder_att()
        self.face_encoder = face_encoder_att()
        self.speech_value_memory = nn.Parameter(torch.FloatTensor(slot_size, slot_channel_size))
        self.face_key_memory = nn.Parameter(torch.FloatTensor(slot_size, slot_channel_size))
        self.slot_size = slot_size
        init.normal_(self.speech_value_memory, mean=0, std=0.5)
        init.normal_(self.face_key_memory, mean=0, std=0.5)


    def forward(self, face, speech, face_attribute=False):
        """
        face [batch_size, dim_shape]
        speech [batch_size, dim_shape]
        """
        self.speech_encoder.eval()
        with torch.no_grad():
            speech_emb = self.speech_encoder(speech)  # [batch_size, 256]
        face_emb = self.face_encoder(face)  # [batch_size, 256]

        N = face.size(0)
        speech_address = F.cosine_similarity(speech_emb.unsqueeze(1), self.speech_value_memory.expand(N,-1,-1), dim=2)
        speech_address_log = F.log_softmax(speech_address, dim=1)
        speech_address = F.softmax(speech_address, dim=1)   # [Batch_size, slot_size]
        speech_emb_recall = torch.matmul(speech_address, self.speech_value_memory)

        face_address = F.cosine_similarity(face_emb.unsqueeze(1), self.face_key_memory.expand(N,-1,-1), dim=2) 
        face_address_log = F.log_softmax(face_address, dim=1)
        face_address = F.softmax(face_address, dim=1)
        face_emb_recall = torch.matmul(face_address, self.speech_value_memory)

        return speech_emb, speech_emb_recall, face_emb, face_emb_recall, speech_address_log, face_address_log


    def forward_speech(self, speech):
        self.speech_encoder.eval()
        with torch.no_grad():
            speech_emb = self.speech_encoder(speech)  # [batch_size, 256]        
        return speech_emb


    def forward_face(self, face, if_return_address=False):
        face_emb = self.face_encoder(face)
        N = face.size(0)
        face_address = F.cosine_similarity(face_emb.unsqueeze(1), self.face_key_memory.expand(N,-1,-1), dim=2) 
        face_address_log = F.log_softmax(face_address, dim=1)
        face_address = F.softmax(face_address, dim=1)
        face_emb_recall = torch.matmul(face_address, self.speech_value_memory)
        if if_return_address:
            return face_emb_recall, face_address_log
        else:
            return face_emb_recall



    def inference_address_face(self, face, face_attribute=False):
        face_emb = self.face_encoder(face)

        if self.if_attribute:
            age_label = face_attribute[:,0]
            gender_label = face_attribute[:,1]
            race_label = face_attribute[:,2]
            age_emb = self.age_attribute_embedding(age_label)
            gender_emb = self.gender_attribute_embedding(gender_label)
            race_emb = self.race_attribute_embedding(race_label)
            face_emb = torch.cat((face_emb, age_emb, gender_emb, race_emb), dim=1)

        N = face.size(0)
        face_address = F.cosine_similarity(face_emb.unsqueeze(1), self.face_key_memory.expand(N,-1,-1), dim=2) 
        face_address = F.softmax(face_address, dim=1)
        return face_address


    def inference_address_speech(self, speech):
        speech_emb = self.speech_encoder(speech)
        N = speech.size(0)
        speech_address = F.cosine_similarity(speech_emb.unsqueeze(1), self.speech_value_memory.expand(N,-1,-1), dim=2)
        speech_address = F.softmax(speech_address, dim=1)
        return speech_address


    def inference(self, face, face_attribute=False):
        face_emb = self.face_encoder(face)

        if self.if_attribute:
            age_label = face_attribute[:,0]
            gender_label = face_attribute[:,1]
            race_label = face_attribute[:,2]
            age_emb = self.age_attribute_embedding(age_label)
            gender_emb = self.gender_attribute_embedding(gender_label)
            race_emb = self.race_attribute_embedding(race_label)
            face_emb = torch.cat((face_emb, age_emb, gender_emb, race_emb), dim=1)

        N = face.size(0)
        face_address = F.cosine_similarity(face_emb.unsqueeze(1), self.face_key_memory.expand(N,-1,-1), dim=2) 
        face_address = F.softmax(face_address, dim=1)
        face_emb_recall = torch.matmul(face_address, self.speech_value_memory)
        return face_emb_recall
    
    
    def edit_inference(self, face_man, face_woman, man_weight):
        face_man_emb = self.face_encoder(face_man)
        face_woman_emb = self.face_encoder(face_woman)


        N = face_man.size(0)
        face_man_address = F.cosine_similarity(face_man_emb.unsqueeze(1), self.face_key_memory.expand(N,-1,-1), dim=2) 
        face_man_address = F.softmax(face_man_address, dim=1)

        face_woman_address = F.cosine_similarity(face_woman_emb.unsqueeze(1), self.face_key_memory.expand(N,-1,-1), dim=2) 
        face_woman_address = F.softmax(face_woman_address, dim=1)

        face_address = man_weight * face_man_address + (1-man_weight) * face_woman_address
        face_emb_recall = torch.matmul(face_address, self.speech_value_memory)
        return face_emb_recall, face_address


    def inference_speech(self, speech, return_recall):
        speech_emb = self.speech_encoder(speech)
        if not return_recall:
            return speech_emb
        else:
            N = speech.size(0)
            speech_address = F.cosine_similarity(speech_emb.unsqueeze(1), self.speech_value_memory.expand(N,-1,-1), dim=2)
            speech_address = F.softmax(speech_address, dim=1) 
            print(speech_address.shape)
            speech_emb_recall = torch.matmul(speech_address, self.speech_value_memory)
            return speech_emb_recall
        

if __name__ == '__main__':
    face_emb = torch.randn(8, 512)
    speech_emb = torch.randn(8, 256)
    model = facevoice_memorynet(48, 256) 
    out = model(face_emb, speech_emb)
    model_2 = facevoice_memorynet_vae(48, 256)
    # speech_emb, speech_emb_recall, face_emb, face_emb_recall, speech_address_log, face_address_log, face_emb_recall_vae, mu, log_sigma
    out = model_2(face_emb, speech_emb)

    


