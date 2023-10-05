import os
import tabnanny

import torch
import torch.nn.functional as F
from thop import profile

from layers import *
import matplotlib
matplotlib.use('Agg')

class EBlock(nn.Module):
    def __init__(self, out_channel, num_res=8):
        super(EBlock, self).__init__()

        layers = [ResBlock(out_channel, out_channel) for _ in range(num_res)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DBlock(nn.Module):
    def __init__(self, channel, num_res=8):
        super(DBlock, self).__init__()

        layers = [ResBlock(channel, channel) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class UNet(nn.Module):
    def __init__(self, in_ch=3, base_ch=32, num_res=2):
        super(UNet, self).__init__()
        self.Encoder = nn.ModuleList([
            EBlock(base_ch, num_res),
            EBlock(base_ch * 2, num_res),
            EBlock(base_ch * 4, num_res),
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_ch*2, num_res),
            DBlock(base_ch, num_res)
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(in_ch, base_ch, kernel_size=3, relu=True, stride=1),
            BasicConv(base_ch * 1, base_ch * 2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_ch * 2, base_ch * 4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_ch * 4, base_ch * 2, kernel_size=3, relu=True, stride=1),
            BasicConv(base_ch * 2, base_ch * 1, kernel_size=3, relu=True, stride=1)
        ])

        self.up1=BasicConv(base_ch * 4, base_ch * 2, kernel_size=4, relu=True, stride=2, transpose=True)
        self.up2=BasicConv(base_ch * 2, base_ch * 1, kernel_size=4, relu=True, stride=2, transpose=True)

    def forward(self, x):
        '''Feature Extract 0'''
        x_ = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)

        '''Down Sample 1'''
        z = self.feat_extract[1](res1)
        res2 = self.Encoder[1](z)

        '''Down Sample 2'''
        z = self.feat_extract[2](res2)
        res3 = self.Encoder[2](z)

        deepz=res3

        '''Up Sample 2'''
        z=self.up1(res3)
        z = self.feat_extract[3](torch.cat([z, res2], dim=1))
        z = self.Decoder[0](z)

        '''Up Sample 1'''
        z=self.up2(z)
        z = self.feat_extract[4](torch.cat([z, res1], dim=1))
        z = self.Decoder[1](z)

        return z, deepz



class INIKNet(nn.Module):
    def __init__(self, config):
        super(INIKNet, self).__init__()
        in_ch=config['in_ch']
        base_ch=config['base_ch']
        num_res_unet=config['num_res_unet']
        max_kernel_size = config['max_kernel_size']
        basis_num = config['basis_num']
        num_res_lstm = config['num_res_lstm']
        learnable_freq = config['learnable_freq']
        w_max = config['w_max']
        w_min = config['w_min']
        self.unet=UNet(in_ch=in_ch, base_ch=base_ch, num_res=num_res_unet)
        group_num=max_kernel_size//2
        self.group_num=group_num
        self.basis_num=basis_num
        # which type?
        self.lstms_basis_select = BiLSTMLayer_stack(hidden_ch=base_ch, num_res=num_res_lstm)
        self.lstms_scale_select = BiLSTMLayer_stack(hidden_ch=base_ch, num_res=num_res_lstm)

        self.scale_select = nn.Conv2d(base_ch, group_num+1, kernel_size=1)
        self.basis_select = nn.Conv2d(base_ch, basis_num, kernel_size=1)  # 1d

        self.inr_conv = SizeGroupINRConvPolar(max_kernel_size=max_kernel_size, num_ch=in_ch,
                                              basis_num=basis_num, w_max=w_max, w_min=w_min,
                                              learnable_freq=learnable_freq)


        self.sum = nn.Conv2d((group_num+1)*basis_num*3, 3, kernel_size=1)



    def forward(self, x):
        x1=F.interpolate(x, scale_factor=0.5, mode='bilinear')
        x2=F.interpolate(x, scale_factor=0.25, mode='bilinear')

        xs=[x2, x1, x]

        unet_outs = []
        for xi in xs:
            unet_outs.append(self.unet(xi)[0])

        lstms_basis_select=self.lstms_basis_select(unet_outs)
        lstms_scale_select=self.lstms_scale_select(unet_outs)

        scale_selects=[]
        for i, select in enumerate(lstms_scale_select):
            select=self.scale_select(select)
            select = torch.softmax(select, dim=1)
            scale_selects.append(select)

        basis_selects=[]
        for i, beta in enumerate(lstms_basis_select):
            b, _, h, w = beta.shape
            beta = self.basis_select(beta) # b, m / (group_num+1)*m, h, w
            basis_selects.append(beta)

        inr_convs=[]
        for xi in xs:
            inr_convs.append(self.inr_conv(xi)) # b, group_num+1, 3*m, h, w

        multiplies=[]
        for inr_conv, scale_selec, basis_select in zip(inr_convs, scale_selects, basis_selects):
            multiplies.append((inr_conv * scale_selec.unsqueeze(2) * basis_select.repeat(1,3,1,1).unsqueeze(1)).flatten(1, 2))  # 1d

        outs=[]
        for multiply, xi in zip(multiplies, xs):
            outs.append(self.sum(multiply)+xi)

        return outs


class SizeGroupINRConvPolar(nn.Module):
    def __init__(self, max_kernel_size=17, num_ch=3, basis_num=5, w_max=7., w_min=1., w_list=None, learnable_freq=False):
        super(SizeGroupINRConvPolar, self).__init__()
        if w_list is None:
            w_list=[w_min+(w_max-w_min)/(basis_num-1)*i for i in range(basis_num)]
            if learnable_freq:
                newwlist=[torch.nn.Parameter(torch.scalar_tensor(w_list[i], dtype=torch.float32)) for i in range(basis_num)]
                w_list=newwlist
        assert len(w_list)==basis_num
        self.w_list=w_list
        self.num_ch=num_ch
        self.kernelINR_list=nn.ModuleList(KernelINR_Polar(hidden_dim=64, w=w_list[i]) for i in range(basis_num))

        self.basis_num=basis_num
        self.max_kernel_size=max_kernel_size
        self.kernel_sizes=[(2*(i+1)+1, 2*(i+1)+1) for i in range(max_kernel_size//2)]
        # self.kernel_sizes = [(max_kernel_size, max_kernel_size) for i in range(max_kernel_size // 2)]

        self.padding=max_kernel_size//2
        self.group_num=len(self.kernel_sizes)

        masks=[] # [1x1, 3x3, ..., 15x15, ...]

        cords=[] # [1x1xc, 3x3xc, ..., 15x15xc, ...]

        empty=torch.zeros(self.basis_num, 1, 1, 1, device='cuda')
        # delta[0, :,max_kernel_size//2, max_kernel_size//2]=1
        delta = torch.ones(self.basis_num, 1, 1, 1, device='cuda')

        self.delta=delta
        self.empty=empty

        for siz in self.kernel_sizes:
            mask = torch.ones(siz, device='cuda', dtype=torch.float32) * (3 ** 2) / (siz[0] * siz[1])
            # mask=torch.ones(siz, device='cuda', dtype=torch.float32)*(max_kernel_size**2)/(siz[0]*siz[1])
            # mask = torch.ones(siz, device='cuda', dtype=torch.float32)
            masks.append(mask)

            cord=shape2polar_coordinate(shape=siz, device='cuda')
            cords.append(cord)

        self.masks=masks
        self.cords=cords


    def forward(self, x):
        b, c, h, w = x.shape
        kernels = []
        maps = []
        for k in range(self.group_num) :
            kernels_g = []
            for i in range(self.basis_num):
                kernel = self.kernelINR_list[i](self.cords[k])  # h w
                kernel = kernel*self.masks[k]
                kernels_g.append(kernel.unsqueeze(0))
            kernels_g=torch.cat(kernels_g, dim=0)  # m h w
            maps_g = F.conv2d(x, kernels_g.repeat(self.num_ch, 1, 1).unsqueeze(1),
                              padding=self.kernel_sizes[k][0]//2,
                              groups=self.num_ch)  # b 3*m h w
            maps.append(maps_g.unsqueeze(1))
            kernels.append(kernels_g)
        maps=torch.cat(maps, dim=1)  # b gn 3*m h w

        null_map=torch.zeros(b, 1, self.num_ch*self.basis_num, h, w, device='cuda')

        maps=torch.cat([null_map, maps], dim=1)  # b gn+1 3*m h w

        return maps



if __name__ == '__main__':

    net = INIKNet(base_ch=32, basis_num=10, max_kernel_size=15, num_res_unet=2, num_res_lstm=2, w_max=12, w_min=2).cuda()
    input = torch.randn(1, 3, 640, 640).cuda()
    flops, params = profile(net, [input])
    print("Number of parameter: %.2fM" % (params / 1e6))
    print(f"FLOPs:{flops / 1e9:.2f}G")





