import os
import tabnanny

import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import cv2
import time
import torch
import torchvision.transforms.functional as F


class Dataset(data.Dataset):
    def __init__(self, img_path, gt_path, crop_size=(256, 256)):
        super(type(self), self).__init__()

        self.crop_size = crop_size
        st = time.time()
        print('loading data')
        img_names=sorted(os.listdir(img_path))
        gt_names=sorted(os.listdir(gt_path))
        self.train_img_list=[]
        self.train_gt_list=[]
        for img_name, gt_name in zip(img_names, gt_names):
            img_name=os.path.join(img_path, img_name)
            gt_name=os.path.join(gt_path, gt_name)
            # self.train_img_list.append(torch.from_numpy(np.array(cv2.imread(img_name)).transpose((2, 0, 1))))
            # self.train_gt_list.append(torch.from_numpy(np.array(cv2.imread(gt_name)).transpose((2, 0, 1))))
            self.train_img_list.append(img_name)
            self.train_gt_list.append(gt_name)
        print('loading data finished', time.time() - st)

    def __len__(self):
        return len(self.train_gt_list)

    def data_augmentation(self, img, crop_left, crop_top, hf, vf, rot):
        img = img[:, crop_top:crop_top + self.crop_size[1], crop_left:crop_left + self.crop_size[0]]
        if hf:
            img = F.hflip(img)
        if vf:
            img = F.vflip(img)
        img = torch.rot90(img, rot, [1, 2])
        return img

    def __getitem__(self, idx):

        # clear_img = self.train_gt_list[idx]
        # blurry_img = self.train_img_list[idx]
        clear_img = torch.from_numpy(np.array(cv2.imread(self.train_gt_list[idx])).transpose((2, 0, 1)))
        blurry_img = torch.from_numpy(np.array(cv2.imread(self.train_img_list[idx])).transpose((2, 0, 1)))
        _, h, w = clear_img.shape
        crop_left = int(np.floor(np.random.uniform(0, w - self.crop_size[0] + 1)))
        crop_top = int(np.floor(np.random.uniform(0, h - self.crop_size[1] + 1)))
        hf = np.random.randint(0, 2)
        vf = np.random.randint(0, 2)
        rot = np.random.randint(0, 4)
        blurry_img = self.data_augmentation(blurry_img, crop_left, crop_top, hf, vf, rot) / 255.
        clear_img = self.data_augmentation(clear_img, crop_left, crop_top, hf, vf, rot) / 255.
        batch = {'img': blurry_img, 'gt': clear_img}
        return batch


class TestDataset_CHUK(data.Dataset):
    def __init__(self,img_path):
        super(type(self), self).__init__()
        st = time.time()
        self.img_names = sorted(os.listdir(img_path))
        self.test_img_list = []
        for img_name in self.img_names:
            img_name = os.path.join(img_path, img_name)
            # self.test_img_list.append(torch.from_numpy(np.array(cv2.imread(img_name)).transpose((2, 0, 1))))
            self.test_img_list.append(img_name)
        print('loading data finished', time.time() - st)

    def __len__(self):
        return len(self.test_img_list)

    def __getitem__(self, idx):
        # blurry_img = self.test_img_list[idx]
        blurry_img = torch.from_numpy(np.array(cv2.imread(self.test_img_list[idx])).transpose((2, 0, 1)))
        # print(blurry_img.shape)
        _, h, w = blurry_img.shape
        aim_h = int(np.floor(h / 16) * 16)
        aim_w = int(np.floor(w / 16) * 16)
        blurry_img = blurry_img[:, :aim_h, :aim_w] / 255.
        # print(blurry_img.shape)
        batch = {'img': blurry_img}
        return batch



class LFDOFDataset(data.Dataset):
    def __init__(self, img_path, crop_size=(256, 256)):
        super(LFDOFDataset, self).__init__()
        st = time.time()

        self.crop_size=crop_size

        self.img_names, self.gt_names = load_LFDOF_file_list(img_path)
        self.train_img_list = []
        self.train_gt_list = []
        for img_name, gt_name in zip(self.img_names, self.gt_names):
            # self.train_img_list.append(torch.from_numpy(np.array(cv2.imread(img_name)).transpose((2, 0, 1))))
            # self.train_gt_list.append(torch.from_numpy(np.array(cv2.imread(gt_name)).transpose((2, 0, 1))))
            self.train_img_list.append(img_name)
            self.train_gt_list.append(gt_name)
        print('loading data finished', time.time() - st)

    def __len__(self):
        return len(self.train_gt_list)

    def data_augmentation(self, img, crop_left, crop_top, hf, vf, rot):
        img = img[:, crop_top:crop_top + self.crop_size[1], crop_left:crop_left + self.crop_size[0]]
        if hf:
            img = F.hflip(img)
        if vf:
            img = F.vflip(img)
        img = torch.rot90(img, rot, [1, 2])
        return img

    def __getitem__(self, idx):

        # clear_img = self.train_gt_list[idx]
        # blurry_img = self.train_img_list[idx]
        clear_img = torch.from_numpy(np.array(cv2.imread(self.train_gt_list[idx])).transpose((2, 0, 1)))
        blurry_img = torch.from_numpy(np.array(cv2.imread(self.train_img_list[idx])).transpose((2, 0, 1)))
        _, h, w = clear_img.shape
        crop_left = int(np.floor(np.random.uniform(0, w - self.crop_size[0] + 1)))
        crop_top = int(np.floor(np.random.uniform(0, h - self.crop_size[1] + 1)))
        hf = np.random.randint(0, 2)
        vf = np.random.randint(0, 2)
        rot = np.random.randint(0, 4)
        blurry_img = self.data_augmentation(blurry_img, crop_left, crop_top, hf, vf, rot) / 255.
        clear_img = self.data_augmentation(clear_img, crop_left, crop_top, hf, vf, rot) / 255.
        batch = {'img': blurry_img, 'gt': clear_img}
        return batch


class TestDataset(data.Dataset):
    def __init__(self,img_path, gt_path):
        super(type(self), self).__init__()
        st = time.time()
        self.img_names = sorted(os.listdir(img_path))
        self.gt_names = sorted(os.listdir(gt_path))
        self.test_img_list = []
        self.test_gt_list = []
        for img_name, gt_name in zip(self.img_names, self.gt_names):
            img_name = os.path.join(img_path, img_name)
            gt_name = os.path.join(gt_path, gt_name)
            # self.test_img_list.append(torch.from_numpy(np.array(cv2.imread(img_name)).transpose((2, 0, 1))))
            # self.test_gt_list.append(torch.from_numpy(np.array(cv2.imread(gt_name)).transpose((2, 0, 1))))
            self.test_img_list.append(img_name)
            self.test_gt_list.append(gt_name)
        print('loading data finished', time.time() - st)

    def __len__(self):
        return len(self.test_img_list)

    def __getitem__(self, idx):
        # clear_img = self.test_gt_list[idx]
        # blurry_img = self.test_img_list[idx]
        clear_img = torch.from_numpy(np.array(cv2.imread(self.test_gt_list[idx])).transpose((2, 0, 1)))
        blurry_img = torch.from_numpy(np.array(cv2.imread(self.test_img_list[idx])).transpose((2, 0, 1)))
        # print(blurry_img.shape)
        _, h, w = blurry_img.shape
        aim_h = int(np.floor(h / 32) * 32)
        aim_w = int(np.floor(w / 32) * 32)
        clear_img = clear_img[:, :aim_h, :aim_w] / 255.
        blurry_img = blurry_img[:, :aim_h, :aim_w] / 255.
        # print(blurry_img.shape)
        batch = {'img': blurry_img, 'gt': clear_img}
        return batch


def load_RTF_file_list(input_path):
    df_img_path_list = []
    gt_img_path_list = []

    df_dirs=sorted(os.listdir(os.path.join(input_path, 'image')))
    print(df_dirs)
    for di in df_dirs:
        df_imgs_name=sorted(os.listdir(os.path.join(input_path, 'image', di)))
        for name in df_imgs_name:
            df_img_path_list.append(os.path.join(input_path, 'image', di, name))
            gt_img_path_list.append(os.path.join(input_path, 'GT','sharp'+name[5:]))
    return df_img_path_list, gt_img_path_list


def load_LFDOF_file_list(path):
    df_img_path_list=[]
    gt_img_path_list=[]

    df_img_dirs=sorted(os.listdir(os.path.join(path, 'input')))

    # print(df_img_dirs)
    for di in df_img_dirs:
        gt_path=os.path.join(path, 'ground_truth', f'{di}.png')
        df_imgs_path_for_gt=os.listdir(os.path.join(path, 'input', di))
        for df_path in df_imgs_path_for_gt:
            df_img_path_list.append(os.path.join(path, 'input', di, df_path))
            gt_img_path_list.append(gt_path)
    return df_img_path_list, gt_img_path_list


class RTFTestDataset(data.Dataset):
    def __init__(self,path):
        super(type(self), self).__init__()
        st = time.time()

        self.img_names, self.gt_names=load_RTF_file_list(path)
        self.test_img_list=[]
        self.test_gt_list = []
        for img_name, gt_name in zip(self.img_names, self.gt_names):
            self.test_img_list.append(torch.from_numpy(np.array(cv2.imread(img_name)).transpose((2, 0, 1))))
            self.test_gt_list.append(torch.from_numpy(np.array(cv2.imread(gt_name)).transpose((2, 0, 1))))
            # self.test_img_list.append(img_name)
            # self.test_gt_list.append(gt_name)
        print('loading data finished', time.time() - st)

    def __len__(self):
        return len(self.test_img_list)

    def __getitem__(self, idx):
        clear_img = self.test_gt_list[idx]
        blurry_img = self.test_img_list[idx]
        # clear_img = torch.from_numpy(np.array(cv2.imread(self.test_gt_list[idx])).transpose((2, 0, 1)))
        # blurry_img = torch.from_numpy(np.array(cv2.imread(self.test_img_list[idx])).transpose((2, 0, 1)))
        # print(blurry_img.shape)
        _, h, w = blurry_img.shape
        aim_h = int(np.floor(h / 16) * 16)
        aim_w = int(np.floor(w / 16) * 16)
        clear_img = clear_img[:, :aim_h, :aim_w] / 255.
        blurry_img = blurry_img[:, :aim_h, :aim_w] / 255.
        # print(blurry_img.shape)
        batch = {'img': blurry_img, 'gt': clear_img}
        return batch
