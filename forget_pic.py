from scipy.stats import spearmanr
import os
import json
import torchvision
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm

from tracin_forget.order_examples import difficult_num



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print("using {} device.".format(device))
data_transform = {
    "train": transforms.Compose([transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    "val": transforms.Compose([transforms.Resize(256),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

# 10000张测试图片
test_set = torchvision.datasets.CIFAR10(root='../datasets', train=False,
                                        download=False, transform=data_transform["val"])
val_num = len(test_set)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1,
                                          shuffle=False, num_workers=0)

train_bar = tqdm(test_loader)

test_eximg = torch.zeros(10, 10, 1 , 3, 224, 224)
label_extract = torch.zeros(10,1).long()
for i in range(10):
    label_extract[i] = i
class_num = [0,0,0,0,0,0,0,0,0,0]   #计数每个类别到了第几个
list_cls_num = [0,0,0,0,0,0,0,0,0,0] #计数找到了几个相关类别的pic
img_num = 0
account = 0
# test_exlabel = torch.zeros(10,1).long()

# torch.flatten(t)
for step, data in enumerate(train_bar):
    images, labels = data
    if(account < len(difficult_num) and difficult_num[account] == step):
        if(list_cls_num[labels[0]] < 10):#list_cls_num[labels[0]] < 10 and
            test_eximg[labels[0]][list_cls_num[labels[0]]] = images
            list_cls_num[labels[0]] += 1
            account += 1
            class_num[labels[0]] += 1
        else:
            account += 1
            class_num[labels[0]] += 1

print("遗忘次数多的数据已导入")



