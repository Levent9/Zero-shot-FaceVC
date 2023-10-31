import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.basic_layers import LinearNorm
import torch.nn.init as init


class face_encoder_base(nn.Module):
    def __init__(self):
        super(face_encoder_base, self).__init__()
        self.linearelu = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU())

    def forward(self,face):
        output = self.linearelu(face)
        return output


class face_encoder_att(nn.Module):
    def __init__(self, dim_emb_in=512, dim_emb_out=256):
        super(face_encoder_att,self).__init__()
        self.query = LinearNorm(dim_emb_in, dim_emb_out)
        self.key = LinearNorm(dim_emb_in, dim_emb_out)
        self.value = LinearNorm(dim_emb_in, dim_emb_out)
        self.softmax  = nn.Softmax(dim=-1)

    def forward(self,x):
        """
            inputs :
                x : input feature maps( dim_emb )
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        x = x.unsqueeze(1)
        # print(x.shape)
        proj_query  = self.query(x)
        proj_key =  self.key(x)
        energy =  torch.matmul(proj_query.permute(0,2,1), proj_key)
        attention = self.softmax(energy)
        proj_value = self.value(x)

        out = torch.matmul(proj_value, attention.permute(0,2,1))
        out = out.squeeze(1)  # [batch_size, fea_dim]
        out = nn.functional.normalize(out, dim=1)
        return out







if __name__ == '__main__':
    input_ = torch.randn(8, 3, 128, 128) # image_size[batch, T, 3, 128, 128]
    model = face_encoder_att_aug()
    output = model(input_)
    # print(output.shape)