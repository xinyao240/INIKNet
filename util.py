import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import termcolor
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM
import lpips
import copy
import torchvision.models as models

class vgg_features(nn.Module):
    def __init__(self):
        super(vgg_features, self).__init__()
        self.net=nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=False),
        nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=False),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=False),
        nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=False),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=False),
        nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=False),
        nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=False),
        nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=False),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=False),
        nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=False),
        nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=False),
        nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=False),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=False),
        nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=False),
        nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=False),
        nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=False),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        )

vgg=models.vgg19(pretrained=True).cuda()
vgg_feats=vgg_features().cuda()
vgg_feats.net.load_state_dict(vgg.features.state_dict())


# loss_fn_alex = lpips.LPIPS(net='vgg').cuda()
loss_fn_alex = lpips.LPIPS(net='alex').cuda()

def vgg_loss(x,y):
    l1 = nn.L1Loss().cuda()
    xs = []
    ys = []
    i_s = [2, 7, 14]

    for i in range(len(vgg_feats.net)):
        x = vgg_feats.net[i](x)
        y = vgg_feats.net[i](y)
        if i in i_s:
            xs.append(x)
            ys.append(y)

    loss=0

    for i in range(3):
        loss+=l1(xs[i], ys[i])

    return loss

def freq_loss(x, y):
    x_fft=torch.fft.fft2(x)
    y_fft=torch.fft.fft2(y)
    real_diff=torch.abs(x_fft.real-y_fft.real).mean()
    imag_diff = torch.abs(x_fft.imag - y_fft.imag).mean()
    loss=real_diff+imag_diff
    return loss

def charbonnier_loss(pred, target, eps=1e-12):
    """Charbonnier loss.

    Args:
        pred (Tensor): Prediction Tensor with shape (n, c, h, w).
        target ([type]): Target Tensor with shape (n, c, h, w).

    Returns:
        Tensor: Calculated Charbonnier loss.
    """
    return torch.sqrt((pred - target)**2 + eps).mean()

def multi_scale_loss(xs, y, mse_lambda=1., freq_lambda=0.1, lpips_lambda=0.2, l1_lambda=0.1, char_lambda=0.1, vgg_lambda=0.1):
    '''

    :param xs: [H//4*W//4, H//2*W//2, H*W]
            y: H*W
    :return:
    '''

    ys=[F.interpolate(y, scale_factor=0.25, mode='bilinear'), F.interpolate(y, scale_factor=0.5, mode='bilinear'), y]
    weights=[1,1,1]

    mse=nn.MSELoss().cuda()
    l1=nn.L1Loss().cuda()

    global loss_fn_alex

    vgg_los = weights[0] * vgg_loss(xs[0], ys[0]) + weights[1] * vgg_loss(xs[1], ys[1]) + weights[
        2] * vgg_loss(xs[2], ys[2])
    char_loss=weights[0]*charbonnier_loss(xs[0], ys[0])+weights[1]*charbonnier_loss(xs[1], ys[1])+weights[2]*charbonnier_loss(xs[2], ys[2])
    l1_loss=weights[0]*l1(xs[0], ys[0])+weights[1]*l1(xs[1], ys[1])+weights[2]*l1(xs[2], ys[2])
    mse_loss=weights[0]*mse(xs[0], ys[0])+weights[1]*mse(xs[1], ys[1])+weights[2]*mse(xs[2], ys[2])
    freqloss=weights[0]*freq_loss(xs[0], ys[0])+weights[1]*freq_loss(xs[1], ys[1])+weights[2]*freq_loss(xs[2], ys[2])
    lpips_loss=weights[0]*loss_fn_alex.forward(xs[0]*2.-1., ys[0]*2.-1.).mean()\
               +weights[1]*loss_fn_alex.forward(xs[1]*2.-1., ys[1]*2.-1.).mean()\
               +weights[2]*loss_fn_alex.forward(xs[2]*2.-1., ys[2]*2.-1.).mean()

    loss=mse_lambda*mse_loss+freq_lambda*freqloss+lpips_lambda*lpips_loss+l1_lambda*l1_loss+char_lambda*char_loss+vgg_lambda*vgg_los

    psnr = 10 * torch.log(1 ** 2 / mse(xs[2], ys[2])) / np.log(10)
    return {
        'loss':loss,
        'mse': mse_loss,
        'freq loss': freqloss,
        'lpips loss': lpips_loss,
        'psnr':psnr
    }


def compute_metrics(out, gt):
    lpips_val=loss_fn_alex.forward(out * 2. - 1., gt * 2. - 1.).mean()
    out_numpy=out.squeeze().cpu().numpy().transpose(1,2,0).clip(0,1)
    gt_numpy=gt.squeeze().cpu().numpy().transpose(1,2,0).clip(0,1)
    psnr = PSNR(gt_numpy, out_numpy)
    ssim = SSIM(gt_numpy, out_numpy, channel_axis=-1, data_range=1)


    return {
        'psnr':psnr,
        'ssim':ssim,
        'lpips': lpips_val.item(),
        'out_numpy':out_numpy
    }


def toRed(content):
    return termcolor.colored(content, "red", attrs=["bold"])


def toGreen(content):
    return termcolor.colored(content, "green", attrs=["bold"])


def toBlue(content):
    return termcolor.colored(content, "blue", attrs=["bold"])


def toCyan(content):
    return termcolor.colored(content, "cyan", attrs=["bold"])


def toYellow(content):
    return termcolor.colored(content, "yellow", attrs=["bold"])


def toMagenta(content):
    return termcolor.colored(content, "magenta", attrs=["bold"])


def toGrey(content):
    return termcolor.colored(content, "grey", attrs=["bold"])


def toWhite(content):
    return termcolor.colored(content, "white", attrs=["bold"])

