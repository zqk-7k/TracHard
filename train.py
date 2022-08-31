import os
import json
import torchvision
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm

from model import resnet34
from tracin_forget.get_dataset import train_img,label_train,test_loader
# from tracin_forget.get_datasetall import train_img,label_train,test_loader

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    net = resnet34()

    # load pretrain weights
    # download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth
    model_weight_path = "../pth_file1/resnet34-pre.pth"
    # model_weight_path = "../pth_50000_pre/resNet34_epoch6.pth"
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    net.load_state_dict(torch.load(model_weight_path, map_location=device))
    # for param in net.parameters():
    #     param.requires_grad = False

    # change fc layer structure
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, 10)
    net.to(device)


    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)

    epochs = 150
    best_acc = 0.0
    save_path = '../pth_file2/resNet34_tracin_' #save_path = './resNet34_new.pth'
    save_path = '../pth_10000_tracin/resNet34_epoch{}.pth'
    rangsize = int(len(train_img)/10)
    rangsize = 1000
    for epoch in range(epochs):
        # train
        net.train()
        time_start = time.perf_counter()
        running_loss = 0.0

        for step in range(rangsize):
            optimizer.zero_grad()
            logits = net(train_img[step * 10:step * 10 + 10].to(device))
            loss = loss_function(logits, label_train[step * 10:step * 10 + 10].to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # validate
        predict_yno = []
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(test_loader)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs, dim=1)[1]
                predict_yno.append(int(predict_y[0]))
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

            val_accurate = acc / 10000
            print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %  # 打印epoch，step，loss，accuracy
                  (epoch + 1, step + 1, running_loss / 500, val_accurate))

            print('%f s' % (time.perf_counter() - time_start))  # 打印耗时
            running_loss = 0.0



        # if val_accurate > best_acc:
        #     best_acc = val_accurate
        #     torch.save(net.state_dict(), save_path.format(epoch))

    print('Finished Training')


if __name__ == '__main__':
    main()
