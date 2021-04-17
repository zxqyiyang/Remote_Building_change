# encoding: utf-8
"""
@author: ZxqYiYang
@file: test.py
@time: 2021/4/14 21:26
"""
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from data import My_Dataset
from torch.utils.data import DataLoader
from model import LinkNet34

def test(model, path, model_path, test_dataset):
    if os.path.exists(path):
        print(path, "文件夹已经存在")
    else:
        os.makedirs(path)
    # device = 'cuda'
    device = "cpu"
    device = torch.device(device)
    net = model.to(device)
    # model.load_state_dict(torch.load(model_path))
    model.load_state_dict(torch.load(model_path, map_location='cpu'))

    net.eval()
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    for i, (img, lab) in enumerate(test_loader):
        # print(i)
        out = net(img.to(torch.float32).to(device=device))
        # out = out.cpu().detach().numpy()
        out = out.detach().numpy()

        # print(out.shape)
        # result = np.argmxax(out, 1)
        # result = np.squeeze(result)
        result = np.array(out[0,1,:,:])*255.*2
        result_1 = np.zeros_like(result)
        result_1[result>240] = 255
        # print(np.max(result), np.min(result))
        # x = input()
        image_save_path = str(path) + str("image_") + str(i) + ".png"
        label_save_path = str(path) + str("label_") + str(i) + ".png"
        result_save_path = str(path) + str("result_") + str(i) + ".png"
        img = img.transpose(3, 1)
        img = img.transpose(2, 1)
        img, lab = np.squeeze(img.numpy())*255., np.squeeze(lab.numpy())*255.
        # print(img.shape, lab.shape)
        cv2.imwrite(image_save_path, cv2.merge([np.squeeze(img[:,:,2]), np.squeeze(img[:,:, 1]), np.squeeze(img[:,:,0])] ))
        cv2.imwrite(label_save_path, cv2.merge([lab[1,:,:]]))
        cv2.imwrite(result_save_path,cv2.merge([result_1]))

if __name__ == "__main__":
    test(model=LinkNet34(), path="./result/result/", model_path="./result/1_199", test_dataset=My_Dataset(mode="test"))
