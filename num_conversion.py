import os

import torch

import time

from torchvision import transforms, datasets

import torchvision
from tqdm import tqdm
from tracin_forget.top50_before import label50

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
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1,
                                          shuffle=False, num_workers=0)

time_start = time.perf_counter()
account = [0,0,0,0,0,0,0,0,0,0]  #到了某类别的哪一个
label50_account = [0,0,0,0,0,0,0,0,0,0]  #到了label50某类别的哪一个



train_bar = tqdm(train_loader)
rank = []
top50 = []

model_weight_path = '../pth_50000_nopre/resNet34_tracin_epoch10.pth'
# net.load_state_dict(torch.load(model_weight_path, map_location=device))
# net.to(device)
for step, data in enumerate(train_bar):
    if (step < 50000):
        images, labels = data
        cls_num = int(labels[0])
        num = int(label50_account[labels[0]])
        if(num < 2500 and account[cls_num] == label50[cls_num][num]):
            label50_account[labels[0]] += 1
            top50.append(step)
        account[cls_num] += 1


print(top50)



