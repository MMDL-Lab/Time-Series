import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Mamba_Enc import Encoder, EncoderLayer
from layers.RevIN import RevIN
import numpy as np
from mamba_ssm import Mamba


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.patch_len = configs.patch_len
        self.use_norm = configs.use_norm
        self.revin_layer = RevIN(configs.enc_in, affine=True, subtract_last=False)
        # Embedding
        self.start_fc = nn.Linear(in_features=1, out_features=configs.d_model)
        self.class_strategy = configs.class_strategy
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    Mamba(d_model=configs.d_model, d_state=16, d_conv=4, expand=2),
                    Mamba(d_model=configs.d_model*configs.patch_len, d_state=16, d_conv=4, expand=2),
                    configs.seq_len,
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                    p_len=configs.patch_len
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.projector = nn.Linear(configs.d_model*configs.seq_len, configs.pred_len, bias=True)

    def FFT_for_Period(self, x, k):
        # [B, T, C]
        xf = torch.fft.rfft(x, dim=1)
        # find period by amplitudes
        frequency_list = abs(xf).mean(0).mean(-1)
        frequency_list[0] = 0
        _, top_list = torch.topk(frequency_list, k)
        top_list = top_list.detach().cpu().numpy()
        period = x.shape[1] // top_list
        return period, abs(xf).mean(-1)[:, top_list]

    def forecast(self, x):
        # period_list, period_weight = self.FFT_for_Period(x, k=5)
        # print(period_list)
        # exit()
        if self.use_norm:
            x = self.revin_layer(x, 'norm')

        B, L, M = x.shape
        x = x.permute(0, 2, 1)
        out = self.start_fc(x.unsqueeze(-1))  # [B, M, L, D]
        out = torch.reshape(out, (out.shape[0]*out.shape[1], out.shape[2], out.shape[3]))  # [b*m, l, d]

        out = self.encoder(out)

        out = torch.reshape(out, (B, M, out.shape[-2], out.shape[-1]))
        out = torch.reshape(out, (out.shape[0], out.shape[1], out.shape[2] * out.shape[3]))

        out = self.projector(out).permute(0, 2, 1)

        if self.use_norm:
            out = self.revin_layer(out, 'denorm')

        return out


    def forward(self, x):
        out = self.forecast(x)
        return out
