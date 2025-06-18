import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import *


class EncoderLayer(nn.Module):
    def __init__(self, mamba1, mamba2, seq_len, d_model, d_ff=None, dropout=0.1, activation="gelu", p_len=16, pe='zeros', learn_pe=True):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.mamba1 = mamba1
        self.mamba2 = mamba2
        self.d_model = d_model
        self.period = p_len
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.w_p1 = positional_encoding(pe, learn_pe, p_len, d_model)
        self.w_p2 = positional_encoding(pe, learn_pe, int(seq_len/p_len), p_len*d_model)
        self.dropout = nn.Dropout(dropout)
        self.attn = nn.Linear(2*d_model, d_model)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x):
        B, L, D = x.shape
        out = x.unfold(dimension=-2, size=self.period, step=self.period)  # [B, P_N, D, P_L]
        out = out.permute(0, 1, 3, 2)
        b, patch_num, patch_len, dim = out.shape
        # loc = torch.reshape(out, (out.shape[0] * out.shape[1], out.shape[2], out.shape[-1]))
        for i in range(patch_num):
            loc_x = out[:, i, :, :]
            loc_x = self.dropout(loc_x + self.w_p1)
            loc_x = self.mamba1(loc_x)  # [b, p_l, d]
            loc_x = loc_x.unsqueeze(-3)
            # loc = torch.reshape(loc, (b, patch_num, loc.shape[-2], loc.shape[-1]))
            if i == 0:
                loc = loc_x
            else:
                loc = torch.cat([loc, loc_x], dim=1)

        glo = torch.reshape(out, (out.shape[0], out.shape[1], out.shape[2] * out.shape[-1]))
        glo = self.dropout(glo + self.w_p2)
        glo = self.mamba2(glo)  # [b, p_n, p_l*d]
        glo = torch.reshape(glo, (b, glo.shape[-2], patch_len, dim))
        # glo = self.attn(glo)
        # glo = glo.unsqueeze(-1)
        # new_x = self.attn_fusion(loc, glo)
        new_x = torch.cat([loc, glo], dim=-1)
        new_x = self.attn(new_x)
        new_x = torch.reshape(new_x, (new_x.shape[0], patch_num * patch_len, dim))

        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y)


class Encoder(nn.Module):
    def __init__(self, mamba_layers, norm_layer):
        super(Encoder, self).__init__()
        self.mamba_layers = nn.ModuleList(mamba_layers)
        self.norm = norm_layer

    def forward(self, x):
        # x [B, L, D]
        for mamba_layer in self.mamba_layers:
            x = mamba_layer(x)

        if self.norm is not None:
            x = self.norm(x)

        return x
