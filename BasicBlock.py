from torch import nn
import torch
import torch.nn.functional as F


class DownSampleBatchNorm(nn.Module):
    def __init__(self, c_in):
        super(DownSampleBatchNorm, self).__init__()
        self.conv = nn.Sequential(
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.BatchNorm1d(c_in),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class UpSampleBatchNorm(nn.Module):
    def __init__(self, c_in, c_out):
        super(UpSampleBatchNorm, self).__init__()
        self.up_conv = nn.Sequential(
            nn.ConvTranspose1d(c_in, c_out, kernel_size=16, stride=2, bias=True, padding=7),
            nn.BatchNorm1d(c_out),
            nn.LeakyReLU()
        )

    def forward(self, x):
        x = self.up_conv(x)
        return x

class ResBlockTwoConv(nn.Module):
    def __init__(self, ni, nf):
        super(ResBlockTwoConv, self).__init__()
        self.convblock1 = nn.Sequential(
            nn.Conv1d(ni, nf, 3, padding='same', padding_mode='reflect'),
            nn.BatchNorm1d(nf),
            nn.LeakyReLU()
        )
        self.convblock2 = nn.Sequential(
            nn.Conv1d(nf, nf, 3, padding='same', padding_mode='reflect'),
            nn.BatchNorm1d(nf),
            nn.LeakyReLU()
        )
        self.shortcut = nn.BatchNorm1d(nf) if ni == nf else nn.Conv1d(ni, nf, 1)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        res = x

        x = self.convblock1(x)
        x = self.convblock2(x)

        x = torch.add(x, self.shortcut(res))
        x = self.act(x)
        return x

class CodeBook_Attention(nn.Module):
    def __init__(self, channel, length):
        super(CodeBook_Attention, self).__init__()
        self.to_k_v = nn.Parameter(torch.rand(1, channel, length))
        self.scale = torch.sqrt(torch.tensor(length).float())

    def forward(self, x):
        q = x
        k = self.to_k_v
        v = self.to_k_v
        energy = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attention = F.softmax(energy, dim=-1)
        out = torch.matmul(attention, v)
        return out


class Res_CodeBook_Attention(nn.Module):
    def __init__(self, channel, length):
        super(Res_CodeBook_Attention, self).__init__()
        self.cba = CodeBook_Attention(channel, length)

    def forward(self, x):
        res = x
        x = self.cba(x)
        return torch.add(x, res)