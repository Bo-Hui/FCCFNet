from regex import D
from multiprocessing import reduction
from typing import Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log, ceil, floor

class MultiDimCA(nn.Module):
    def __init__(self,channel,reduction=16,dim=8) -> None:
        super().__init__()
        #把图像压缩到dim*dim的大小，对齐最低特征图维度
        self.maxpool=nn.AdaptiveMaxPool2d(dim)#(B,C,H,W)-->(B,C,dim,dim)
        self.avgpool=nn.AdaptiveAvgPool2d(dim)#(B,C,H,W)-->(B,C,dim,dim)

        #两个分组卷积对通道描述符进行细粒度的处理
        self.max_conv = nn.Conv2d(channel,channel,kernel_size=dim,groups=channel,bias=False)
        self.avg_conv = nn.Conv2d(channel,channel,kernel_size=dim,groups=channel,bias=False)

        self.se = nn.Sequential(
            nn.Conv2d(channel,channel//reduction,1,bias=False),
            nn.ReLU(),
            nn.Conv2d(channel//reduction,channel,1,bias=False)
        )

        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x):
        #先对x进行pooling
        max_result=self.maxpool(x) # 通过最大池化压缩全局空间信息: (B,C,H,W)--> (B,C,dim,dim)
        avg_result=self.avgpool(x) # 通过平均池化压缩全局空间信息: (B,C,H,W)--> (B,C,dim,dim)
        
        #通过分组卷积对通道描述符进行细粒度的处理
        max_result = self.max_conv(max_result) # 共享同一个分组卷积: (B,C,dim,dim)--> (B,C,dim,dim)
        avg_result = self.avg_conv(avg_result) # 共享同一个分组卷积: (B,C,dim,dim)--> (B,C,dim,dim)

        #最后激励
        max_out = self.se(max_result) # 共享同一个MLP: (B,C,dim,dim)--> (B,C,1,1)
        avg_out = self.se(avg_result) # 共享同一个MLP: (B,C,dim,dim)--> (B,C,1,1)
 
        #最后通过sigmoid获得注意力权重
        output = self.sigmoid(max_out+avg_out) # 相加,然后通过sigmoid获得权重:(B,C,1,1)

        return x*output # 与输入相乘,获得注意力权重后的特征图:(B,C,H,W)

# 改进方案2：使用门控融合机制替代简单拼接
class CGAFusion(nn.Module):
    def __init__(self, in_channel, out_channel, reduction=1):
        super(CGAFusion, self).__init__()
        # 对上一阶段的特征图降维
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

        self.ca = MultiDimCA(out_channel, reduction)

        # 门控机制
        self.gate_x = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Sigmoid()
        )
        
        self.gate_y = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Sigmoid()
        )
        
        # 混合建模
        self.conv_mix = nn.Sequential(
            nn.Conv2d(out_channel*2, out_channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, y):
        y = self.conv_in(y)  # 对上一阶段的
        y = self.ca(y)  # 应用通道注意力
        
        # 门控机制
        gate_x = self.gate_x(x)
        gate_y = self.gate_y(y)
        
        # 加权融合
        fused = torch.cat([gate_x * x, gate_y * y], dim=1)
        
        # 混合建模
        result = self.conv_mix(fused)

        return result

class ConvBNReLU(nn.Module):

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()

        padding = kernel_size // 2 if dilation == 1 else dilation
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 先卷积，再 BN，再 ReLU
        x = self.relu(self.bn(self.conv(x)))

        return x

class DownConvBNReLU(ConvBNReLU):
    
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1, flag: bool = True):
        super().__init__(in_ch, out_ch, kernel_size, dilation)
        self.down_flag = flag

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.down_flag:
            x = F.max_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)

        x = self.relu(self.bn(self.conv(x)))

        return x

class UpConvBNReLU(ConvBNReLU):

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1, flag: bool = True):
        super().__init__(in_ch, out_ch, kernel_size, dilation)
        self.up_flag = flag

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        if self.up_flag:
            x1 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=False)
        return self.relu(self.bn(self.conv(torch.cat([x1, x2], dim=1))))


class RSU(nn.Module):
    """ Residual U-block """

    def __init__(self, height: int, in_ch: int, mid_ch: int, out_ch: int):
        super().__init__()

        assert height >= 2
        self.conv_in = ConvBNReLU(in_ch, out_ch)  # stem
        encode_list = [DownConvBNReLU(out_ch, mid_ch, flag=False)]
        decode_list = [UpConvBNReLU(mid_ch * 2, mid_ch, flag=False)]
        for i in range(height - 2):
            encode_list.append(DownConvBNReLU(mid_ch, mid_ch))
            decode_list.append(UpConvBNReLU(mid_ch * 2, mid_ch if i < height - 3 else out_ch))
        encode_list.append(ConvBNReLU(mid_ch, mid_ch, dilation=2))

        self.encode_modules = nn.ModuleList(encode_list)
        self.decode_modules = nn.ModuleList(decode_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = self.conv_in(x)

        x = x_in
        encode_outputs = []
        for m in self.encode_modules:
            x = m(x)
            encode_outputs.append(x)

        x = encode_outputs.pop()
        for m in self.decode_modules:
            x2 = encode_outputs.pop()
            x = m(x, x2)

        return x + x_in

class RSU4F(nn.Module):

    def __init__(self, in_ch: int, mid_ch: int, out_ch: int):
        super().__init__()
        self.conv_in = ConvBNReLU(in_ch, out_ch)
        self.encode_modules = nn.ModuleList([ConvBNReLU(out_ch, mid_ch),
                                             ConvBNReLU(mid_ch, mid_ch, dilation=2),
                                             ConvBNReLU(mid_ch, mid_ch, dilation=4),
                                             ConvBNReLU(mid_ch, mid_ch, dilation=8)])

        self.decode_modules = nn.ModuleList([ConvBNReLU(mid_ch * 2, mid_ch, dilation=4),
                                             ConvBNReLU(mid_ch * 2, mid_ch, dilation=2),
                                             ConvBNReLU(mid_ch * 2, out_ch)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = self.conv_in(x)

        x = x_in
        encode_outputs = []
        for m in self.encode_modules:
            x = m(x)
            encode_outputs.append(x)

        x = encode_outputs.pop()
        for m in self.decode_modules:
            x2 = encode_outputs.pop()
            x = m(torch.cat([x, x2], dim=1))

        return x + x_in


class DBFF(nn.Module):

    def __init__(self, in_channel, kernel_size=1, stride=1):
        super(DBFF, self).__init__()

        self.in_channel = in_channel
        self.inter_channel = in_channel
        self.out_channel = in_channel * 2
        self.kernel_size = kernel_size
        self.stride = stride

        self.conv_decoder_in = nn.Sequential(
            nn.Conv2d(self.in_channel, self.inter_channel, kernel_size=1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(self.inter_channel),
            nn.ReLU(True))

        self.conv_encoder_in = nn.Sequential(
            nn.Conv2d(self.in_channel, self.inter_channel, kernel_size=1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(self.inter_channel),
            nn.ReLU(True))

        #解码器输出采用空间注意力
        self.conv_decoder_up = nn.Sequential(
            nn.Conv2d(self.inter_channel, self.in_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.in_channel),
            nn.ReLU(True)
        )

        self.conv_encoder_up = nn.Sequential(
            nn.Conv2d(self.inter_channel, self.in_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.in_channel),
            nn.ReLU(True)
        )

        self.softmax = nn.Softmax(dim=0)

        self.out = nn.Sequential(
            nn.Conv2d(self.in_channel, self.out_channel, 3, 1, 1),
            nn.BatchNorm2d(self.out_channel),
            nn.ReLU(True)
        )

        self.encoder_ca = MultiDimCA(self.inter_channel,reduction=1)
        self.decoder_ca = MultiDimCA(self.inter_channel,reduction=1)

    def forward(self, D, E):
        #先计算decoder支路
        D_ = self.conv_decoder_in(D)
        D_ = self.decoder_ca(D_)#通过通道注意力
        
        E_ = self.conv_encoder_in(E)
        E_ = self.encoder_ca(E_)#通过通道注意力
        
        #两个支路通过conv_up
        D_ = self.conv_decoder_up(D_)
        E_ = self.conv_encoder_up(E_)

        out_add = D_+E_
        out_mul = D_*E_

        mul_mask = (out_mul > 0).float()
        mul_mask = torch.max_pool2d(mul_mask, kernel_size=2, stride=2)
        mul_mask = torch.nn.functional.interpolate(mul_mask, size=D_.shape[2:], mode='nearest')

        out = out_add*mul_mask + out_add

        out = self.out(out)

        return out

class Net(nn.Module):

    def __init__(self, cfg: dict, out_ch: int = 1):
        super().__init__()
        assert "encode" and "decode" in cfg
        self.encode_num = len(cfg["encode"])
        encode_list = []
        side_list = []
        se_list = []
        #channel_down_conv = []
        last_c = 0
        for c in cfg["encode"]:
            # c: [height, in_ch, mid_ch, out_ch, RSU4F, side]
            assert len(c) >= 6
            encode_list.append(RSU(*c[:4]) if c[4] is False else RSU4F(*c[1:4]))
            se_list.append(CGAFusion(last_c,c[3],reduction=1))#输入输出通道大小一致，所以对齐
            last_c = c[3]

            if c[5] is True:
                side_list.append(nn.Conv2d(c[3], out_ch, kernel_size=3, padding=1))
        self.encode_modules = nn.ModuleList(encode_list)
        self.se_modules = nn.ModuleList(se_list)
        

        decode_list = []
        dbff_list = []
        for i, c in enumerate(cfg["decode"]):
            # c: [height, in_ch, mid_ch, out_ch, RSU4F, side]
            assert len(c) >= 6
            decode_list.append(RSU(*c[:4]) if c[4] is False else RSU4F(*c[1:4]))
            dbff_list.append(DBFF(int(c[1] / 2)))

            if c[5] is True:
                side_list.append(nn.Conv2d(c[3], out_ch, kernel_size=3, padding=1))

        self.decode_modules = nn.ModuleList(decode_list)
        self.side_modules = nn.ModuleList(side_list)
        
        self.out_conv = nn.Conv2d(self.encode_num * out_ch, out_ch, kernel_size=1)

        self.dbff_modules = nn.ModuleList(dbff_list)
        self.bn = nn.BatchNorm2d(out_ch)


    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        _, _, h, w = x.shape

        encode_outputs = []
        for i, m in enumerate(self.encode_modules):
            x_out = m(x)#编码器输出
            if i == 0:
                x = x_out
            else:
                #不需要对x进行下采样，因为已经采样过了
                #x = F.max_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)
                #将x_out和x进行concat
                #x = torch.concat([x_out, x], dim=1)
                #对x使用通道注意力
                x = self.se_modules[i](x_out,x)
                
            encode_outputs.append(x)
            if i != self.encode_num - 1:
                x = F.max_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)

        x = encode_outputs.pop()
        decode_outputs = [x]
        for m in zip(self.decode_modules, self.dbff_modules):
            x2 = encode_outputs.pop()
            x = F.interpolate(x, size=x2.shape[2:], mode='bilinear', align_corners=False)
            x = m[1](x, x2)
            x = m[0](x)
            decode_outputs.insert(0, x)

        side_outputs = []

        #简单上采样
        for m in self.side_modules:
            x = decode_outputs.pop()
            x = F.interpolate(m(x), size=[h, w], mode='bilinear', align_corners=False)
            side_outputs.insert(0, x)

        x = self.out_conv(torch.concat(side_outputs, dim=1))

        if self.training:
            # do not use torch.sigmoid for amp safe
            return [x] + side_outputs
        else:
            return self.bn(x)

def FCCFNet_S(out_ch: int = 1):
    cfg = {
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "encode": [[7, 3, 4, 8, False, False],  # En1
                   [6, 8, 4, 8, False, False],  # En2
                   [5, 8, 4, 8, False, False],  # En3
                   [4, 8, 4, 8, False, False],  # En4
                   [4, 8, 4, 8, True, False],  # En5
                   [4, 8, 4, 8, True, True]],  # En6
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "decode": [[4, 16, 4, 8, True, True],  # De5
                   [4, 16, 4, 8, False, True],  # De4
                   [5, 16, 4, 8, False, True],  # De3
                   [6, 16, 4, 8, False, True],  # De2
                   [7, 16, 4, 8, False, True]]  # De1
    }

    return Net(cfg, out_ch)
