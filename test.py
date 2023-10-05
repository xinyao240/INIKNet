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
model_name=model_config['model_name']

net=INIKNet(model_config).cuda()
net.load_state_dict(torch.load(f'checkpoints/{model_name}.pth'))
# evaluate
save_out_dir=f'out/{model_name}'
os.makedirs(save_out_dir, exist_ok=True)
print('start evaluating ...')
net.eval()

with open('test_config.json','r') as file:
    strr = file.read()
    test_config = json.loads(strr)

test_data_rtf=TestDataset(img_path=test_config['rtf']['img_path'],
                          gt_path=test_config['rtf']['gt_path'])
test_data_dpdd=TestDataset(img_path=test_config['dpdd']['img_path'],
                          gt_path=test_config['dpdd']['gt_path'])
test_data_realdof=TestDataset(img_path=test_config['realdof']['img_path'],
                          gt_path=test_config['realdof']['gt_path'])

test_loader_dpdd=DataLoader(test_data_dpdd, shuffle=False, batch_size=1)
test_loader_realdof=DataLoader(test_data_realdof, shuffle=False, batch_size=1)
test_loader_rtf=DataLoader(test_data_rtf, shuffle=False, batch_size=1)


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