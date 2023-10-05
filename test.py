from model import INIKNet
import torch
from util import multi_scale_loss
from data import Dataset, TestDataset
import numpy as np
import random
from torch.utils.data import DataLoader
from tqdm import tqdm
from util import toRed, toBlue, toCyan, toGreen, toYellow, compute_metrics
import os
import cv2
import json

with open('model_config.json','r') as file:
    strr = file.read()
    model_config = json.loads(strr)

test_data_rtf=TestDataset(img_path='/home/lab535/data/yx/Data/RTFDataset/image/0',
                          gt_path='/home/lab535/data/yx/Data/RTFDataset/GT')
test_data_dpdd=TestDataset(img_path='/home/lab535/data/yx/Data/dd_dp_dataset_png/test_c/source',
                           gt_path='/home/lab535/data/yx/Data/dd_dp_dataset_png/test_c/target')
test_data_realdof=TestDataset(img_path='/home/lab535/data/yx/Data/RealDOF/source',
                              gt_path='/home/lab535/data/yx/Data/RealDOF/target')

test_loader_dpdd=DataLoader(test_data_dpdd, shuffle=False, batch_size=1)
test_loader_realdof=DataLoader(test_data_realdof, shuffle=False, batch_size=1)
test_loader_rtf=DataLoader(test_data_rtf, shuffle=False, batch_size=1)
model_name='INIKNet'

num_res_unet=2
num_res_lstm=2
basis_num=10
max_kernel_size=15
w_max=12
w_min=2
learnable_freq=True

net=INIKNet(model_config).cuda()
net.load_state_dict(torch.load(f'checkpoints/{model_name}.pth'))
# evaluate
save_out_dir=f'out/{model_name}'
os.makedirs(save_out_dir, exist_ok=True)
print('start evaluating ...')
net.eval()

save_img=False
# dpdd
save_out_dataset_dir=os.path.join(save_out_dir, 'dpdd')
os.makedirs(save_out_dataset_dir, exist_ok=True)
f = open(f'{save_out_dataset_dir}/metrics.txt', 'w')

sum_psnr = []
sum_ssim = []
sum_lpips = []
for i, batch in enumerate(tqdm(test_loader_dpdd)):
    # for k in batch:
    #     batch[k] = batch[k].to(device)
    gt = batch['gt'].cuda()
    img = batch['img'].cuda()

    with torch.no_grad():
        out = net(img)

        metrics = compute_metrics(out[-1], gt)

    psnr=metrics['psnr']
    ssim=metrics['ssim']
    lpips_val=metrics['lpips']
    sum_ssim.append(metrics['ssim'])
    sum_psnr.append(metrics['psnr'])
    sum_lpips.append(metrics['lpips'])
    img_name=test_data_dpdd.img_names[i]
    save_p = os.path.join(save_out_dataset_dir, test_data_dpdd.img_names[i])
    if save_img:
        cv2.imwrite(save_p, np.clip(metrics['out_numpy'], 0,1)*255)


    f.write(f'{img_name} psnr/ssim/lpips {psnr}/{ssim}/{lpips_val}\n')


avg_ssim = sum(sum_ssim) / len(sum_ssim)
avg_psnr = sum(sum_psnr) / len(sum_psnr)
avg_lpips = sum(sum_lpips) / len(sum_lpips)
print(f'dpdd avg val ssim:{toBlue(str(avg_ssim)),} psnr:{toGreen(str(avg_psnr))} lpips:{toRed(str(avg_lpips))}')
f.write(f'dpdd avg val ssim:{toBlue(str(avg_ssim)),} psnr:{toGreen(str(avg_psnr))} lpips:{toRed(str(avg_lpips))}')
f.close()
torch.cuda.empty_cache()

# realdof
sum_psnr = []
sum_ssim = []
sum_lpips = []
save_out_dataset_dir=os.path.join(save_out_dir, 'realdof')
os.makedirs(save_out_dataset_dir, exist_ok=True)
f = open(f'{save_out_dataset_dir}/metrics.txt', 'w')

for i, batch in enumerate(tqdm(test_loader_realdof)):
    # for k in batch:
    #     batch[k] = batch[k].to(device)
    gt = batch['gt'].cuda()
    img = batch['img'].cuda()

    with torch.no_grad():
        out = net(img)

        metrics = compute_metrics(out[-1], gt)

    psnr = metrics['psnr']
    ssim = metrics['ssim']
    lpips_val = metrics['lpips']
    sum_ssim.append(metrics['ssim'])
    sum_psnr.append(metrics['psnr'])
    sum_lpips.append(metrics['lpips'])
    img_name = test_data_realdof.img_names[i]
    save_p = os.path.join(save_out_dataset_dir, test_data_realdof.img_names[i])
    if save_img:
        cv2.imwrite(save_p, np.clip(metrics['out_numpy'], 0, 1) * 255)


    f.write(f'{img_name} psnr/ssim/lpips {psnr}/{ssim}/{lpips_val}\n')


avg_ssim = sum(sum_ssim) / len(sum_ssim)
avg_psnr = sum(sum_psnr) / len(sum_psnr)
avg_lpips = sum(sum_lpips) / len(sum_lpips)
print(f'realdof avg val ssim:{toBlue(str(avg_ssim))} psnr:{toGreen(str(avg_psnr))} lpips:{toRed(str(avg_lpips))}')
f.write(f'realdof avg val ssim:{toBlue(str(avg_ssim))} psnr:{toGreen(str(avg_psnr))} lpips:{toRed(str(avg_lpips))}')
f.close()

torch.cuda.empty_cache()

# rtf
sum_psnr = []
sum_ssim = []
sum_lpips = []
save_out_dataset_dir=os.path.join(save_out_dir, 'rtf')
os.makedirs(save_out_dataset_dir, exist_ok=True)
f = open(f'{save_out_dataset_dir}/metrics.txt', 'w')

for i, batch in enumerate(tqdm(test_loader_rtf)):
    # for k in batch:
    #     batch[k] = batch[k].to(device)
    gt = batch['gt'].cuda()
    img = batch['img'].cuda()

    with torch.no_grad():
        out = net(img)

        metrics = compute_metrics(out[-1], gt)

    psnr = metrics['psnr']
    ssim = metrics['ssim']
    lpips_val = metrics['lpips']
    sum_ssim.append(metrics['ssim'])
    sum_psnr.append(metrics['psnr'])
    sum_lpips.append(metrics['lpips'])
    img_name = test_data_rtf.img_names[i]
    save_p = os.path.join(save_out_dataset_dir, test_data_rtf.img_names[i])
    if save_img:
        cv2.imwrite(save_p, np.clip(metrics['out_numpy'], 0, 1) * 255)


    f.write(f'{img_name} psnr/ssim/lpips {psnr}/{ssim}/{lpips_val}\n')


avg_ssim = sum(sum_ssim) / len(sum_ssim)
avg_psnr = sum(sum_psnr) / len(sum_psnr)
avg_lpips = sum(sum_lpips) / len(sum_lpips)
print(f'rtf avg val ssim:{toBlue(str(avg_ssim))} psnr:{toGreen(str(avg_psnr))} lpips:{toRed(str(avg_lpips))}')
f.write(f'rtf avg val ssim:{toBlue(str(avg_ssim))} psnr:{toGreen(str(avg_psnr))} lpips:{toRed(str(avg_lpips))}')
f.close()