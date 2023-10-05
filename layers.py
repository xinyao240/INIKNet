import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=False, norm=False, relu=True,
                 transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias,
                          padding_mode='reflect'))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.ReLU(inplace=False))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x):
        return self.main(x) + x



class StackResBlock(nn.Module):
    def __init__(self,out_channel,num_res=2):
        super(StackResBlock, self).__init__()
        layers = [ResBlock(out_channel, out_channel) for _ in range(num_res)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class LSTMCell(nn.Module):
    def __init__(self, hidden_ch=64, num_res=1):
        super().__init__()
        self.hidden_ch=hidden_ch
        self.layers=StackResBlock(hidden_ch*2, num_res=num_res)
        self.conv=nn.Conv2d(hidden_ch*2, hidden_ch*4, kernel_size=(1,1), padding=0)

    def forward(self, fea, hidden_state):
        if hidden_state is None:
            hidden_state=self.init_state(fea.shape[0], (fea.shape[-2], fea.shape[-1]))
        h, c=hidden_state
        combined=torch.cat([fea,h], dim=1)
        A=self.conv(self.layers(combined))
        (ai, af, ao, ag) = torch.split(A, self.hidden_ch, dim=1)
        i = torch.sigmoid(ai)
        f = torch.sigmoid(af)
        o = torch.sigmoid(ao)
        g = torch.tanh(ag)

        next_c = f * c + i * g
        next_h = o * torch.tanh(next_c)

        return next_h, next_c

    def init_state(self, batch_size, shape):
        return (torch.zeros(batch_size, self.hidden_ch, shape[0], shape[1]).cuda(),
                torch.zeros(batch_size, self.hidden_ch, shape[0], shape[1]).cuda())


class LSTMLayer(nn.Module):
    def __init__(self, hidden_ch=64, direction=0, num_res=1):
        super(LSTMLayer, self).__init__()
        assert direction in [0, 1]  # coarse2fine or fine2coarse
        self.direction = direction
        self.cell = LSTMCell(hidden_ch=hidden_ch, num_res=num_res)

    def forward(self, feature_list):
        '''

        :param feature_list: [f0 (H//4*W//4),f1 (H//2*W//2),f2 (H*W)]
        :return:
        '''
        if self.direction==1:
            feature_list.reverse()
        hs=[]
        cs=[]
        for i in range(len(feature_list)):
            x=feature_list[i]
            if len(hs)==0:
                h, c=self.cell(x, None)
            else:
                if self.direction==1:
                    h, c=self.cell(x,
                                   (F.interpolate(hs[-1], scale_factor=0.5, mode='bilinear'),F.interpolate(cs[-1], scale_factor=0.5, mode='bilinear')))
                else:
                    h, c = self.cell(x,
                                     (F.interpolate(hs[-1], scale_factor=2, mode='bilinear'), F.interpolate(cs[-1], scale_factor=2, mode='bilinear')))

                # if self.direction==1:
                #     h, c=self.cell(x,
                #                    (F.interpolate(hs[-1], scale_factor=0.5),F.interpolate(cs[-1], scale_factor=0.5)))
                # else:
                #     h, c = self.cell(x,
                #                      (F.interpolate(hs[-1], scale_factor=2), F.interpolate(cs[-1], scale_factor=2)))
            hs.append(h)
            cs.append(c)

        if self.direction==1:
            hs.reverse()
            feature_list.reverse()

        return hs




def plus_list(A, B):
    outs=[]
    for a, b in zip(A, B):
        outs.append(a+b)
    return outs

def forward_list(zs, f):
    outs=[]
    for z in zs:
        outs.append(f(z))
    return outs

class LayerNorm2d(nn.Module):
    def __init__(self, dim):
        super(LayerNorm2d, self).__init__()
        self.norm=nn.LayerNorm(dim)
    def forward(self,x):
        '''

        :param x: b c h w
        :return:
        '''
        x=x.permute(0,2,3,1)
        x=self.norm(x)
        x=x.permute(0,3,1,2)
        return x

class BiLSTMLayer_stack(nn.Module):
    def __init__(self, hidden_ch=64, num_res=1):
        super().__init__()
        self.left2right=LSTMLayer(hidden_ch=hidden_ch, direction=0, num_res=num_res)
        self.right2left = LSTMLayer(hidden_ch=hidden_ch, direction=1, num_res=num_res)
        self.norm=LayerNorm2d(hidden_ch)

    def forward(self, xs):
        '''

        Args:
            xs: [f0 (H//4*W//4),f1 (H//2*W//2),f2 (H*W)]
        Returns:

        '''
        h1 = self.right2left(forward_list(xs, self.norm))
        h2 = self.left2right(h1)
        out=plus_list(xs, h2)
        return out

class Sine(nn.Module):
    def __init__(self, w0 = 1.):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)

class Square(nn.Module):
    def forward(self, x):
        return torch.square(x)

class Abs(nn.Module):
    def forward(self,x):
        return torch.abs(x)

class Exp(nn.Module):
    def __init__(self, w=0.5):
        super(Exp, self).__init__()
        self.w=w
    def forward(self, x):
        return torch.exp(self.w*x)

class Shift(nn.Module):
    def forward(self, x):
        x = x - x.min(dim=-2, keepdim=True).min(dim=-3, keepdim=True)
        return x

def shape2coordinate(shape=(3,3), device='cuda', normalize_range=(0.,1.)):
    h, w=shape
    x=torch.arange(0, h, device=device)
    y=torch.arange(0, w, device=device)

    x, y=torch.meshgrid(x, y)

    min, max=normalize_range
    x=x/(h-1)*(max-min)+min
    y=y/(w-1)*(max-min)+min
    cord=torch.cat([x.unsqueeze(-1), y.unsqueeze(-1)], dim=-1)

    return cord



def shape2polar_coordinate(shape=(3,3), device='cuda'):
    h, w=shape
    x=torch.arange(0, h, device=device)
    y=torch.arange(0, w, device=device)

    x, y=torch.meshgrid(x, y)

    min=-1
    max=1
    x=x/(h-1)*(max-min)+min
    y=y/(w-1)*(max-min)+min
    cord=x+1j*y


    r=torch.abs(cord)/np.sqrt(2)
    theta=torch.angle(cord)
    theta_code=torch.cat([(torch.cos(theta).unsqueeze(-1)+1)/2, (torch.sin(theta).unsqueeze(-1)+1)/2], dim=-1)

    cord=torch.cat([r.unsqueeze(-1), theta_code], dim=-1)

    return cord




class KernelINR_Polar(nn.Module):
    def __init__(self, hidden_dim=64, w=1.):
        super().__init__()
        self.layers=nn.Sequential(
            nn.Linear(3, hidden_dim),
            Sine(w),
            nn.Linear(hidden_dim, hidden_dim),
            Sine(w),
            nn.Linear(hidden_dim, 1, bias=False),
        )

    def forward(self, cord):
        k=self.layers(cord).squeeze(-1)
        return k

