from model import INIKNet
import torch
from util import multi_scale_loss
from data import Dataset, TestDataset
import numpy as np
import random
from torch.utils.data import DataLoader
from tqdm import tqdm
from util import toRed, toBlue, toCyan, toGreen, toYellow, compute_metrics
import json

with open('model_config.json','r') as file:
    strr = file.read()
    model_config = json.loads(strr)

with open('train_config.json','r') as file:
    strr = file.read()
    train_config = json.loads(strr)

def seed_everything(seed):
    if seed >= 10000:
        raise ValueError("seed number should be less than 10000")
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0
    seed = (rank * 100000) + seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


seed = 777
seed_everything(seed)
crop_size=train_config["crop_size"]
dataset_n=train_config['dataset']
assert dataset_n in ['dpdd', 'lfdof']

train_data=Dataset(img_path=train_config['img_path'],
                   gt_path=train_config['gt_path'], crop_size=(train_config['crop_size'], train_config['crop_size']))


train_loader=DataLoader(train_data, shuffle=True, batch_size=train_config['batch_size']//train_config['acc_step'], num_workers=train_config['num_workers'])

net=INIKNet(model_config).cuda()
net.train()
model_name=model_config['model_name']
f = open(f'{model_name}.txt', 'a')
f.write(f'\n\n')
start_epoch=train_config['start_epoch']
epoch_n=train_config['total_epochs']-start_epoch
sav_freq=train_config['sav_freq']

optimizer=torch.optim.Adam(net.parameters(), lr=train_config['lr'])
scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer, train_config['mile_stone'], train_config['rate'])

start_from_trained = train_config['start_from_trained']
if start_from_trained:
    net.load_state_dict(torch.load(train_config['weight_path']))
    optimizer.load_state_dict(torch.load(train_config['optimizer_path']))
    scheduler.load_state_dict(torch.load(train_config['scheduler_path']))

for epoch in range(epoch_n):
    sum_los = []
    sum_psnr = []
    sum_freq_loss=[]

    with tqdm(total=len(train_loader),
              desc=f'epoch{start_epoch + epoch + 1}/{start_epoch + epoch_n} train', unit='it', ncols=150) as pbar:
        for i, batch in enumerate(train_loader):
            gt = batch['gt'].cuda()
            img = batch['img'].cuda()

            out = net(img)

            los = multi_scale_loss(out, gt, mse_lambda=1., l1_lambda=0., freq_lambda=0.1, lpips_lambda=0.1, char_lambda=0.)

            los["loss"].backward()

            pbar.set_postfix(
                {
                    'bat_loss': toBlue(f'{los["loss"].item():.5f}'),
                    'learning rate': toYellow(f'{optimizer.param_groups[0]["lr"]}'),
                    'mse_loss': f'{los["mse"].item():.5f}',
                    'freq_loss': f'{los["freq loss"].item():.5f}',
                    'psnr': f'{los["psnr"].item():.5f}'
                 }
            )

            sum_los.append(los["loss"].item())
            sum_psnr.append(los['psnr'].item())
            sum_freq_loss.append(los['freq loss'].item())

            if ((i + 1) % train_config['acc_step']) == 0:
                optimizer.step()  
                optimizer.zero_grad()

            pbar.update(1)

    scheduler.step()

    epoch_avg_train_loss = sum(sum_los) / len(sum_los)
    print('epoch{0}/{1} avg train loss:{2}'.format(
        start_epoch + epoch + 1, start_epoch + epoch_n, epoch_avg_train_loss
    ))
    epoch_avg_train_psnr = sum(sum_psnr) / len(sum_psnr)
    print(f'epoch{start_epoch + epoch + 1}/{start_epoch + epoch_n} train psnr:{epoch_avg_train_psnr}')
    epoch_avg_train_freq_loss = sum(sum_freq_loss) / len(sum_freq_loss)
    print(f'epoch{start_epoch + epoch + 1}/{start_epoch + epoch_n} train freq loss:{epoch_avg_train_freq_loss}')

    f = open(f'{model_name}.txt', 'a')
    msg=f'epoch{start_epoch + epoch + 1}/{start_epoch + epoch_n} avg train loss:{epoch_avg_train_loss}'
    msg+=f'train psnr:{epoch_avg_train_psnr}'
    msg+=f'train freq loss:{epoch_avg_train_freq_loss}\n'
    f.write(msg)
    f.close()


    if (start_epoch + epoch + 1) % sav_freq == 0:
        torch.save(net.state_dict(), f'checkpoints/{model_name}-epoch{start_epoch + epoch + 1}.pth')

    else:
        torch.save(net.state_dict(),
                   f'checkpoints/{model_name}.pth')
        torch.save(optimizer.state_dict(),
                   f'checkpoints/Adam-{model_name}.pth')
        torch.save(scheduler.state_dict(),
                   f'checkpoints/scheduler-{model_name}.pth')




