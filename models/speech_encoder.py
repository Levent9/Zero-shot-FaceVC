import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.basic_layers import LinearNorm, Repara
# from basic_layers import LinearNorm
import torch.nn.init as init

class speech_encoder_base(nn.Module):
    def __init__(self, dim_emb_in = 256, dim_emb_out = 256):
        super(speech_encoder_base, self).__init__()
        self.linearelu = nn.Sequential(
                nn.Linear(dim_emb_in, 256),
                nn.ReLU(),
                nn.Linear(256, dim_emb_out),
                nn.ReLU())

    def forward(self,face):
        output = self.linearelu(face)
        return output


class speech_encoder_att(nn.Module):
    def __init__(self, dim_emb_in=256, dim_emb_out=256):
        super(speech_encoder_att,self).__init__()

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


class speech_encoder_att_reapara(nn.Module):
    def __init__(self, dim_emb_in=256, dim_emb_out=256):
        super(speech_encoder_att_reapara,self).__init__()

        self.query = LinearNorm(dim_emb_in, dim_emb_out)
        self.key = LinearNorm(dim_emb_in, dim_emb_out)
        self.value = LinearNorm(dim_emb_in, dim_emb_out)
        self.softmax  = nn.Softmax(dim=-1)
        self.reparam = Repara(dim_emb_out, 320, dim_emb_out)
        

    def forward(self,x):
        """
            inputs :P
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
        out = nn.functional.normalize(out, dim=1) # with or without normalization
        out, _, _ = self.reparam(out)

        return out


class speech_encoder_att_reapara_linear(nn.Module):
    def __init__(self, dim_emb_in=256, dim_emb_out=256):
        super(speech_encoder_att_reapara_linear,self).__init__()

        self.query = LinearNorm(dim_emb_in, dim_emb_out)
        self.key = LinearNorm(dim_emb_in, dim_emb_out)
        self.value = LinearNorm(dim_emb_in, dim_emb_out)
        self.softmax  = nn.Softmax(dim=-1)
        self.reparam = Repara(dim_emb_out, 320, dim_emb_out)
        self.linear_layer = nn.Linear(dim_emb_out, dim_emb_out)

    def forward(self,x):
        """
            inputs :P
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
        out = nn.functional.normalize(out, dim=1) # with or without normalization
        out, _, _ = self.reparam(out)
        out = self.linear_layer(out)
        return out


class style_token_layer(nn.Module):
    """
    input --- [N, E//2]
    """
    def __init__(self, token_num, emb_dim, num_heads):
        super().__init__()
        self.embed = nn.Parameter(torch.FloatTensor(token_num, emb_dim // num_heads))
        d_q = emb_dim 
        d_k = emb_dim // num_heads
        self.attention = MultiHeadAttention(query_dim = d_q, key_dim = d_k, num_units= emb_dim, num_heads= num_heads)
        init.normal_(self.embed, mean=0, std=0.5)
    
    def forward(self, inputs):
        N = inputs.size(0)
        query = inputs.unsqueeze(1) # [N, 1, emb_dim]
        keys = F.tanh(self.embed).unsqueeze(0).expand(N, -1, -1)  # [N, token_num, E // num_heads]
        style_embed = self.attention(query, keys).squeeze(1)
        style_embed = nn.functional.normalize(style_embed, dim=1)
        return style_embed


class MultiHeadAttention(nn.Module):
    '''
    input:
        query --- [N, T_q, query_dim]
        key --- [N, T_k, key_dim]
    output:
        out --- [N, T_q, num_units]
    '''

    def __init__(self, query_dim, key_dim, num_units, num_heads):

        super().__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.key_dim = key_dim

        self.W_query = nn.Linear(in_features=query_dim, out_features=num_units, bias=False)
        self.W_key = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)
        self.W_value = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)

    def forward(self, query, key):
        querys = self.W_query(query)  # [N, T_q, num_units]
        keys = self.W_key(key)  # [N, T_k, num_units]
        values = self.W_value(key)

        split_size = self.num_units // self.num_heads
        querys = torch.stack(torch.split(querys, split_size, dim=2), dim=0)  # [h, N, T_q, num_units/h]
        keys = torch.stack(torch.split(keys, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]
        values = torch.stack(torch.split(values, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]

        # score = softmax(QK^T / (d_k ** 0.5))
        scores = torch.matmul(querys, keys.transpose(2, 3))  # [h, N, T_q, T_k]
        scores = scores / (self.key_dim ** 0.5)
        scores = F.softmax(scores, dim=3)

        # out = score * V
        out = torch.matmul(scores, values)  # [h, N, T_q, num_units/h]
        out = torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, num_units]

        return out



SPEECH_ENCODER = {
    'base': speech_encoder_base,
    'attention': speech_encoder_att,
    'style_token':style_token_layer,
    'attention_repara':speech_encoder_att_reapara,
    'attention_repara_linear':speech_encoder_att_reapara_linear
}