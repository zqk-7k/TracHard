import os

import torch
import torch.nn as nn
import math
import time
from model import resnet34
from torchvision import transforms, datasets

import torchvision
from tqdm import tqdm
from tracin_forget.top50 import top50

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_transform = {
    "train": transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    "val": transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}


train_set = torchvision.datasets.CIFAR10(root='../datasets', train=True,
                                         download=False, transform=data_transform["train"])
# 加载训练集，实际过程需要分批次（batch）训练
train_loader = torch.utils.data.DataLoader(train_set, batch_size=1,
                                           shuffle=False, num_workers=0)

test_set = torchvision.datasets.CIFAR10(root='../datasets', train=False,
                                            download=False, transform=data_transform["val"])
test_loader = torch.utils.data.DataLoader(test_set, batch_size=10,
                                          shuffle=False, num_workers=0)

time_start = time.perf_counter()


train_bar = tqdm(train_loader)

train_img = torch.zeros(25000, 3, 224, 224)
label_train = torch.zeros(25000).long()

num = 0
for step, data in enumerate(train_bar):
    if (step < 50000):
        if(num < 25000 and step == top50[num]):
            images, labels = data
            train_img[num] = images[0]
            label_train[num] = labels[0]
            num += 1

print(num)








