import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import argparse
import sys
import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from utils import plot_attn

gridI = 20
gridJ = 20

#####################################################################
class CNN0(nn.Module):
    def __init__(self, feature_size, size):
        super().__init__()
        self.k1convL1 = nn.Conv2d(1, int(100/4), size-1, stride=1) 
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.k1convL1(x)
        x = self.activation(x)
        x = torch.reshape(x, (x.size()[0], x.size()[1]*x.size()[2]*x.size()[3]))
        return x

class CNN1(nn.Module):
    def __init__(self, feature_size):
        super().__init__()
        self.k1convL1 = nn.Conv2d(1, 1, 1, stride=1)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.k1convL1(x)
        x = self.activation(x)
        x = torch.reshape(x, (x.size()[0], x.size()[1]*x.size()[2]*x.size()[3]))
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, lookback=24, size=3, max_len=5000):
        super(PositionalEncoding, self).__init__()       
        pe = torch.zeros(max_len, d_model)
        self.lookback = lookback
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        #pe = pe.unsqueeze(0).transpose(0, 1) #B, d_model, seq_len
        pe.requires_grad = True
        self.register_buffer('pe', pe)

        self.cnn = CNN0(d_model, size)
        self.cnn1 = CNN1(d_model)

    def forward(self, x, x1):
        # x1 [B, seq_len, 1]
        # x [B, seq_len, K, K]
        x = torch.reshape(x, (x.size()[0]*x.size()[1], x.size()[2], x.size()[3]))
        x = x.unsqueeze(1) # [B * seq_len, 1, K, K]   
        x = self.cnn(x)
        x = torch.reshape(x, (int(x.size()[0]/self.lookback), self.lookback, x.size()[1]))

        x1 = torch.reshape(x1, (x1.size()[0]*x1.size()[1], x1.size()[2]))
        x1 = x1.unsqueeze(2)
        x1 = x1.unsqueeze(1)
        x1 = self.cnn1(x1)
        x1 = torch.reshape(x1, (int(x1.size()[0]/self.lookback), self.lookback, x1.size()[1]))

        tmp = self.pe[:x.size(1), :].unsqueeze(0) # [1, seq_len, d_model]
        return x + tmp, x1 + tmp

#########################################################
import torch 
from torch import nn
import torch.nn.functional as f
import numpy as np 
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, feature_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.feature_dim = feature_dim
        d_Q = d_K = d_V = feature_dim
            
    
        self.d_K0 = feature_dim // self.num_heads
        self.W_q = nn.Linear(d_Q, feature_dim, bias=False)
        self.W_k = nn.Linear(d_K, feature_dim, bias=False)
        self.W_v = nn.Linear(d_V, feature_dim, bias=False)
        
        self.W_h = nn.Linear(feature_dim, feature_dim)
        self.W_h1 = nn.Linear(feature_dim, feature_dim) 
 
    def scaled_dot_product_attention(self, Q, K, V, attn_mask):
        batch_size = Q.size(0) 
        k_length = K.size(-2) 
        #print(Q.shape)
        #B, Nt, E = Q.shape
        #Q = Q / np.sqrt(E) 
     
        Q = Q / np.sqrt(self.d_K0)                         
        scores = torch.matmul(Q, K.transpose(2,3))          
        scores += attn_mask.unsqueeze(0).unsqueeze(0)
        A = nn.Softmax(dim=-1)(scores)   
        
        H1 = torch.matmul(A, V)     
        H = torch.matmul(A, Q)
        return H, H1, A 

    def split_heads(self, x, batch_size):
        return x.view(batch_size, -1, self.num_heads, self.d_K0).transpose(1, 2)

    def forward(self, X_q, X_k, X_v, attn_mask):
        batch_size, seq_length, dim = X_q.size()

        Q = self.split_heads(self.W_q(X_q), batch_size)  
        K = self.split_heads(self.W_k(X_k), batch_size) 
        V = self.split_heads(self.W_v(X_v), batch_size)  
        
        H_cat, H_cat1, A = self.scaled_dot_product_attention(Q, K, V, attn_mask)
     
        H_cat = H_cat.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_K0) # (bs, q_length, dim)
        H_cat1 = H_cat1.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_K0)
 
        H = self.W_h(H_cat)         
        H1 = self.W_h1(H_cat1)
        return H, H1, A

class LinearTransform(nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        super().__init__()
        self.layer1 = nn.Linear(feature_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, feature_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, feature_dim, num_heads, hidden_dim):
        super().__init__()

        self.mha = MultiHeadAttention(feature_dim, num_heads)
        self.lt = LinearTransform(feature_dim, hidden_dim)
        self.lt1 = LinearTransform(feature_dim, hidden_dim)
        self.layernorm1 = nn.LayerNorm(normalized_shape=feature_dim, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(normalized_shape=feature_dim, eps=1e-6)
    
    def forward(self, x, x1, attn_mask):    
        # x: query, key
        # x1: value        
        attn_output, attn_output1, a = self.mha(x, x, x1, attn_mask) 
        oa = self.layernorm1(x + attn_output) 
        ob = self.lt(oa)  
        oc = self.layernorm2(oa + ob) 

        oa1 = self.layernorm1(x1 + attn_output1)
        ob1 = self.lt1(oa1)
        oc1 = self.layernorm2(oa1 + ob1)

        return oc, oc1, a

class Encoder(nn.Module):
    def __init__(self, num_layers, feature_dim, num_heads, hidden_dim):
        super().__init__()

        self.feature_dim = feature_dim
        self.num_layers = num_layers

        self.enc_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.enc_layers.append(EncoderLayer(feature_dim, num_heads, hidden_dim))
        
    def forward(self, x, x1, attn_mask):

        for i in range(self.num_layers):
            x, x1, a = self.enc_layers[i](x, x1, attn_mask)

        return x1, a  

####################################################################################
class STModel(nn.Module):
    def __init__(self, feature_size=100, num_layers=3, emb_dim=20, lookback=24, size=3):
        super(STModel, self).__init__()

        self.lookback = lookback
        self.sigmoid = nn.Sigmoid() 

        self.transformer_decoder = nn.Linear(feature_size, 1)

        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size, lookback=self.lookback, size=size)
        self.transformer_encoder = Encoder(num_layers=num_layers, feature_dim=feature_size, num_heads=10, hidden_dim=128)
        print('lookback:{0}, num_layer:{1}, feature_size:{2}, emb_dim:{3}, size:{4}'.format(lookback, num_layers, feature_size, emb_dim, size)) 
        self.layer3 = nn.Linear(emb_dim*2 + feature_size, 100)
        self.layer6 = nn.Linear(100, 1)

        self.feature_size= feature_size

        self.emb_dow = nn.Embedding(7, emb_dim) #7 days a week
        self.emb_loc = nn.Embedding(gridI*gridJ, emb_dim) 
        self.emb_time = nn.Embedding(13, emb_dim)
        self.emb_nta = nn.Embedding(200, emb_dim, padding_idx=0)
        #self.emb_nta.weight.requires_grad=False
     
    def forward(self, src, src1, xdow, gid, tid, nid):

        device = 'cuda'

        #Embedding
        dow_emb = self.emb_dow(xdow)
        loc_emb = self.emb_loc(gid)
        time_emb = self.emb_time(tid)
        nta_emb = self.emb_nta(nid)
 
        #transformer embedding time series data
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            mask = self._generate_square_subsequent_mask(src.size()[1]).to(device)
            self.src_mask = mask
 

        src, src1 = self.pos_encoder(src, src1) 
        trans_output, last_a = self.transformer_encoder(src, src1, self.src_mask)
        output = torch.mean(trans_output, dim=1).squeeze()
        output = torch.cat((output, dow_emb, loc_emb), 1)
        output = F.relu(self.layer3(output))
        output = self.layer6(output)
        output = torch.tanh(output)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
