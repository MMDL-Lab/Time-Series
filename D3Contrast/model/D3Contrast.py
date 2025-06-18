import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch_dct as dct
from einops import rearrange
from .attn import FullAttention, AttentionLayer
from .embed import DataEmbedding, TokenEmbedding
from .RevIN import RevIN
from tkinter import _flatten
from mamba_ssm import Mamba


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, self.kernel_size - 1 - math.floor((self.kernel_size - 1) // 2), 1)
        end = x[:, -1:, :].repeat(1, math.floor((self.kernel_size - 1) // 2), 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x):
        for attn_layer in self.attn_layers:
            fre = attn_layer(x, x, x)
        return fre



class D3Contrast(nn.Module):
    def __init__(self, win_size, enc_in, c_out, n_heads=1, d_model=256, e_layers=3, channel=55, d_ff=512, dropout=0.1, activation='gelu', output_attention=True, kernel_size=25, factor=5):
        super(DCdetector, self).__init__()
        self.output_attention = output_attention
        self.channel = channel
        self.win_size = win_size
        self.kernel_size = kernel_size
        self.decomp = series_decomp(self.kernel_size)
        self.trend = Mamba(d_model=enc_in, d_state=win_size, d_conv=4, expand=2)
        self.TE = Mamba(d_model=d_model, d_state=win_size, d_conv=4, expand=2)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

        self.embedding_window_size1 = DataEmbedding(enc_in, d_model, dropout)
        self.embedding_window_size2 = DataEmbedding(enc_in, d_model, dropout)
        self.norm = nn.LayerNorm(d_model)
        # Dual Attention Encoder
        self.fre_enc = Encoder(
            [
                AttentionLayer(
                        FullAttention(factor, attention_dropout=dropout,
                                      output_attention=output_attention), d_model, n_heads)
                     for l in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model)
        )

        self.projection1 = nn.Linear(d_model, c_out, bias=True)
        self.projection2 = nn.Linear(d_model, c_out, bias=True)


    def forward(self, x):
        B, L, M = x.shape #Batch win_size channel
        revin_layer = RevIN(num_features=M)

        # Instance Normalization Operation
        x = revin_layer(x, 'norm')
        seasonal, trend = self.decomp(x)
        trend = self.trend(trend)
        x_new = self.embedding_window_size1(seasonal)  # B L D
        x = self.embedding_window_size2(x)
        
        time_ori = self.TE(x)
        time_flip = self.TE(x.flip(-2))
        time = time_ori + time_flip
        time = self.dropout(self.activation(time))
        time = self.norm(time)
        time = self.projection1(time)
        
        fre = dct.dct(x_new)
        fre, _ = self.fre_enc(fre)
        fre = dct.idct(fre)
        fre = self.projection2(fre)
        fre = fre + trend

        if self.output_attention:
            return time, fre
        else:
            return None
        

