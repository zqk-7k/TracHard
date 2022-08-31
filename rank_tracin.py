import os

import torch
import torch.nn as nn
import math
import time
from model import resnet34
from torchvision import transforms, datasets
from pif.influence_functions_new import get_gradient,tracin_get
from torch.autograd import grad
from data_get import dataset_get,dataset_category_get
import torchvision
from tqdm import tqdm
from tracin_forget.forget_pic import test_eximg,label_extract

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
label10 = []
label20 = []
label40 = []
label80 = []



net = resnet34()
in_channel = net.fc.in_features
net.fc = nn.Linear(in_channel, 10)
net.to(device)

#############################################


# define loss function
loss_function = nn.CrossEntropyLoss()

train_bar = tqdm(train_loader)
rank = []


model_weight_path = '../pth_50000_nopre/resNet34_tracin_epoch10.pth'
net.load_state_dict(torch.load(model_weight_path, map_location=device))
net.to(device)
classnum = 0
for classnum in range(10):
    score_list = []
    account = 0
    label50 = []
    for step, data in enumerate(train_bar):
        if (step < 50000):
            images, labels = data
            logits_train = net(images.to(device))
            loss_train = loss_function(logits_train, labels.to(device))
            grad_z_train = grad(loss_train, net.parameters())
            grad_z_train = get_gradient(grad_z_train, net)

            if(labels[0] != classnum):
                a=1
            else:
                for i in range(10):
                    logits_test = net(test_eximg[classnum][i].to(device))
                    loss_test = loss_function(logits_test, label_extract[classnum].to(device))
                    grad_z_test = grad(loss_test, net.parameters())
                    grad_z_test = get_gradient(grad_z_test, net)
                    score = tracin_get(grad_z_test, grad_z_train)
                    if (i == 0):
                        score_list.append(float(score))
                    else:
                        score_list[account] = score_list[account] + float(score)
                account += 1
            if (step == 49999):
                print(len(score_list))
                print('account:',account)
            if (step == 49999):
                print('进入')
                # score_list存储tracin得分组
                print(len(score_list))

                # score_list_copy用于对数组重新排序，得到排序后结果
                score_list_copy = []

                for p in range(len(score_list)):
                    score_list_copy.append(score_list[p])
                score_list_copy.sort()
                # for k in range(len(score_list)):
                #     if(score_list[k] >= score_list_copy[1000]):
                #         label80.append(k)
                #         if(score_list[k] >= score_list_copy[3000]):
                #             label40.append(k)
                #             if (score_list[k] >= score_list_copy[4000]):
                #                 label20.append(k)
                #                 if (score_list[k] >= score_list_copy[4500]):
                #                     label10.append(k)
                for k in range(len(score_list)):
                    if (score_list[k] >= score_list_copy[2500]):
                        label50.append(k)

    print('%f s' % (time.perf_counter() - time_start))  # 打印耗时
    print(label50)





