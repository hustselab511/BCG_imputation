import torch
from torch import nn

from BasicBlock import DownSampleBatchNorm, UpSampleBatchNorm, ResBlockTwoConv, Res_CodeBook_Attention


class BCGECGConversionModel(nn.Module):
    def __init__(self):
        super(BCGECGConversionModel, self).__init__()
        self.res_block1 = ResBlockTwoConv(1, 64)

        self.down1 = DownSampleBatchNorm(64)
        self.res_block2 = ResBlockTwoConv(64, 128)

        self.down2 = DownSampleBatchNorm(128)
        self.res_block3 = ResBlockTwoConv(128, 256)

        self.down3 = DownSampleBatchNorm(256)
        self.res_block4 = ResBlockTwoConv(256, 512)

        self.down4 = DownSampleBatchNorm(512)
        self.res_block5 = ResBlockTwoConv(512, 512)

        self.down5 = DownSampleBatchNorm(512)
        self.res_block6 = ResBlockTwoConv(512, 512)

        self.cba_bottom = Res_CodeBook_Attention(512, 16)

        self.up6 = UpSampleBatchNorm(512, 512)
        self.cba6 = Res_CodeBook_Attention(512, 32)
        self.up_res_block6 = ResBlockTwoConv(1024, 512)

        self.up5 = UpSampleBatchNorm(512, 512)
        self.cba5 = Res_CodeBook_Attention(512, 64)
        self.up_res_block5 = ResBlockTwoConv(1024, 512)

        self.up4 = UpSampleBatchNorm(512, 256)
        self.cba4 = Res_CodeBook_Attention(512, 128)
        self.up_res_block4 = ResBlockTwoConv(512, 256)

        self.up3 = UpSampleBatchNorm(256, 128)
        self.cba3 = Res_CodeBook_Attention(512, 256)
        self.up_res_block3 = ResBlockTwoConv(256, 128)

        self.up2 = UpSampleBatchNorm(128, 64)
        self.cba2 = Res_CodeBook_Attention(512, 512)
        self.up_res_block2 = ResBlockTwoConv(128, 64)

        self.Conv_1x1 = nn.Conv1d(64, 1, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        x1 = self.res_block1(x)
        x2 = self.down1(x1)
        x2 = self.res_block2(x2)

        x3 = self.down2(x2)
        x3 = self.res_block3(x3)

        x4 = self.down3(x3)
        x4 = self.res_block4(x4)

        x5 = self.down4(x4)
        x5 = self.res_block5(x5)

        encoder_latent = self.down5(x5)
        x6 = self.res_block6(encoder_latent)

        decoder_latent = self.cba_bottom(x6)

        # decoding + concat path
        d6 = self.up6(decoder_latent)
        x5 = self.cba6(x5)

        d6 = torch.cat((x5, d6), 1)
        d6 = self.up_res_block6(d6)

        d5 = self.up5(d6)

        x4 = self.cba5(x4)
        d5 = torch.cat((x4, d5), 1)
        d5 = self.up_res_block5(d5)

        d4 = self.up4(d5)
        x3 = self.cba4(x3)
        d4 = torch.cat((x3, d4), 1)
        d4 = self.up_res_block4(d4)

        d3 = self.up3(d4)
        x2 = self.cba3(x2)
        d3 = torch.cat((x2, d3), 1)
        d3 = self.up_res_block3(d3)

        d2 = self.up2(d3)
        x1 = self.cba2(x1)
        d2 = torch.cat((x1, d2), 1)
        d2 = self.up_res_block2(d2)

        d1 = self.Conv_1x1(d2)
        return d1, encoder_latent, decoder_latent


class BCGECGAlignModel(nn.Module):
    def __init__(self):
        super(BCGECGAlignModel, self).__init__()
        self.bcg2ecg_model = BCGECGConversionModel()
        self.ecg2bcg_model = BCGECGConversionModel()

    def forward(self, bcg_data_tensor, ecg_data_tensor):
        output_ecg_data_tensor, bcg_encoder_latent, ecg_decoder_latent = self.bcg2ecg_model(bcg_data_tensor)
        output_bcg_data_tensor, ecg_encoder_latent, bcg_decoder_latent = self.ecg2bcg_model(ecg_data_tensor)
        return (output_ecg_data_tensor, bcg_encoder_latent, ecg_decoder_latent,
                output_bcg_data_tensor, ecg_encoder_latent, bcg_decoder_latent)