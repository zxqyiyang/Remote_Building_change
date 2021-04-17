# encoding: utf-8
"""
@author: ZxqYiYang
@file: train.py
@time: 2021/4/14 20:21
"""
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data import My_Dataset
from model import LinkNet34

class train_model():
    def __init__(self):
        self.batch_size = 8
        self.epochs = 200
        self.lr = 1e-5
        self.device = "cuda"
        self.acc = []
        self.loss= []
        self.h, self.w = 512, 512

    def save_txt(self, data, path):
        with open(path, "w") as f:
            for i in data:
                f.write(str(i))
                f.write("\n")
        f.close()

    def __call__(self, model, path, dataset):
        if os.path.exists(path):
            print(path, "文件夹已经存在")
        else:
            os.makedirs(path)
        # try:
        device = torch.device(self.device)
        net = model.to(device)
        # except:
        #     net = model.to("cpu")
        net.train()

        criterion = nn.BCELoss()
        optimizer = optim.Adam(net.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        # optimizer = optim.SGD(net.parameters(), lr=self.lr)

        print("======================模型加载成功=======================")
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        print("共有{:4d}个数据参与训练".format(len(train_loader.dataset)))
        for epoch in range(self.epochs):
            train_acc = 0.0
            train_loss= 0.0
            for batch_id, (img, lab) in enumerate(train_loader):
                img, lab = img.to(torch.float32), torch.squeeze(lab.to(torch.float))
                img, lab = img.to(device), lab.to(device)
                out = net(img)
                loss= criterion(out, lab)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                prediction = torch.argmax(out, dim=1)
                lab = torch.argmax(lab, dim=1)
                train_acc += (prediction == lab).sum().float() /lab.shape[1]/ lab.shape[2] / self.batch_size
                train_loss+= float(loss.item())
                torch.cuda.empty_cache()
            print("第{:3d}/{:3d}批次，损失函数：{:.5f}， acc:{:.5f}".format(epoch, self.epochs, train_loss/float((batch_id+1)), train_acc/(batch_id+1)))
            self.loss.append(train_loss/float((batch_id+1)))
            self.acc.append(train_acc/(batch_id+1))
            if (epoch+1) % 20==0 or epoch==(self.epochs-1):
                torch.save(net.state_dict(), str(path)+ str("1_")  + str(epoch))
        print("===============正在保存损失值和正确率=====================")
        self.save_txt(self.loss,path + "loss.txt")
        self.save_txt(self.acc, path + "acc.txt")

if __name__ == "__main__":
    train_function = train_model()
    train_function(model=LinkNet34(num_classes=2), dataset=My_Dataset(), path="./result/")
