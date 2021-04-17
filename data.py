# encoding: utf-8
"""
@author: ZxqYiYang
@file: data.py
@time: 2021/4/14 19:11
"""
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2

class process_data():
    def __init__(self):
        self.width = 512

    def determine(self, label):
        label = np.array(label)
        label_sum = np.sum(label) / np.max(label)
        if label_sum >= 10000:
            mode = True
        else:
            mode = False
        return mode

    def file(self, path):
        if os.path.exists(path):
            print(path, "文件夹已经存在")
        else:
            os.makedirs(path)

    def split_data(self, mode="train"):
        """
        此函数目的是将1024*1024的图像分割成512*512的小图像
        :param mode:
        :return:
        """
        a_save_path = "./data/" + mode + str("/split_A/")
        b_save_path = "./data/" + mode + str("/split_B/")
        l_save_path = "./data/" + mode + str("/split_label/")
        self.file(a_save_path)
        self.file(b_save_path)
        self.file(l_save_path)

        names = os.listdir("./data/" + mode + str("/label/"))  # 获取所有图像名
        for name in tqdm(names):
            img_a_path = "./data/" + mode + str("/A/") + name
            img_b_path = "./data/" + mode + str("/B/") + name
            lab_name = "./data/" + mode + str("/label/") + name
            img_a, img_b, lab = Image.open(img_a_path), Image.open(img_b_path), Image.open(lab_name)
            img_a, img_b, lab = np.array(img_a), np.array(img_b), np.array(lab)
            # print(img_a.shape, img_b.shape, lab.shape)
            for i in range(2):
                for j in range(2):
                    a = img_a[i * 512:(i + 1) * 512, j * 512:(j + 1) * 512, :]
                    b = img_b[i * 512:(i + 1) * 512, j * 512:(j + 1) * 512, :]
                    l = lab[i * 512:(i + 1) * 512, j * 512:(j + 1) * 512]
                    # print(a.shape, b.shape, l.shape)
                    if self.determine(l):
                        cv2.imwrite(a_save_path + str(name).split(".")[0] + "_" + str(i) + str(j) + ".png", cv2.merge([a[:,:,2], a[:,:,1], a[:,:,0]]))
                        cv2.imwrite(b_save_path + str(name).split(".")[0] + "_" + str(i) + str(j) + ".png", cv2.merge([b[:,:,2], b[:,:,1], b[:,:,0]]))
                        cv2.imwrite(l_save_path + str(name).split(".")[0] + "_" + str(i) + str(j) + ".png", cv2.merge([l[:,:]]))

# x = process_data()
# x.split_data(mode="val")

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

class My_Dataset(Dataset):
    def __init__(self, mode="train"):
        self.image_a_path = "./data/" + mode + "/split_A/" #图片A的路径
        self.image_b_path = "./data/" + mode + "/split_B/" #图片B的路径
        self.image_l_path = "./data/" + mode + "/split_label/"#图片Label路径
        self.names = os.listdir(self.image_a_path) # 获取所有的图像名称
        # print(self.names)

    def make_label(self, lab):
        one_zero_label = np.zeros((lab.shape[0], lab.shape[1]))
        one_zero_label[lab>0] = 1
        return one_zero_label

    def __getitem__(self, item):
        img_a = Image.open(self.image_a_path + self.names[item])
        img_b = Image.open(self.image_b_path + self.names[item])
        label = Image.open(self.image_l_path + self.names[item])
        img_a, img_b = np.array(img_a)/255., np.array(img_b)/255.
        label = self.make_label(np.array(label))
        img_a, img_b, label = torch.from_numpy(img_a), torch.from_numpy(img_b), torch.from_numpy(label).to(torch.long)
        label = torch.nn.functional.one_hot(label, num_classes=2)
        img_a, img_b, label = img_a.transpose(0, 2).transpose(1, 2), img_b.transpose(0, 2).transpose(1, 2), label.transpose(0, 2).transpose(1, 2)
        # print(img_a.shape, img_b.shape, label.shape)
        image = img_b - img_a
        # image = torch.cat([img_a, img_b], dim=0)
        return image, label

    def __len__(self):
        return len(self.names)

if __name__ == "__main__":
    train_dataset = My_Dataset(mode="train")
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    for batch_id, (img, lab) in enumerate(train_loader):
        print(img.shape, lab.shape)
        print(torch.max(img), torch.min(img))
        x = input()
        x = 1