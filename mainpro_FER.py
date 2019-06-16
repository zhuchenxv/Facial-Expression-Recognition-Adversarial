'''Train Fer2013 with PyTorch.'''
# 10 crop for data enhancement
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import transforms as transforms
import numpy as np
import os
import argparse
import utils
from fer import FER2013
from torch.autograd import Variable
from models import *
from adversarialbox.train import adv_train, FGSM_train_rnd
from adversarialbox.attacks import FGSMAttack, LinfPGDAttack
from adversarialbox.utils import to_var, pred_batch, test,attack_over_test_data

parser = argparse.ArgumentParser(description='PyTorch Fer2013 CNN Training')
parser.add_argument('--model', type=str, default='VGG19', help='CNN architecture')
parser.add_argument('--dataset', type=str, default='FER2013', help='CNN architecture')
parser.add_argument('--bs', default=128, type=int, help='learning rate')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--adv_train',default=0,type=int)
parser.add_argument('--adv_test',default=0,type=int)
parser.add_argument('--test',default=0,type=int)
#parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
opt = parser.parse_args()

use_cuda = torch.cuda.is_available()

PublicTest_acc_list = []
PrivateTest_acc_list = []
PublicTest_adv_acc_list = []
PrivateTest_adv_acc_list = []
Loss_list = []

learning_rate_decay_start = 80  # 50
learning_rate_decay_every = 5 # 5
learning_rate_decay_rate = 0.9 # 0.9

cut_size = 44
total_epoch = 250

#path = os.path.join(opt.dataset + '_' + opt.model)
if opt.adv_train == 1:
    path = os.path.join(opt.dataset + '_' + opt.model + '_adv')
else:
    path = os.path.join(opt.dataset + '_' + opt.model)
print(path)
if not os.path.isdir(path):
    os.mkdir(path)
# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.Resize((48,48)),
    transforms.RandomCrop(44),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    #transforms.TenCrop(cut_size),
    transforms.ToTensor(),
    #transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])

trainset = FER2013(split = 'Training', transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.bs, shuffle=True, num_workers=1)
PublicTestset = FER2013(split = 'PublicTest', transform=transform_test)
PublicTestloader = torch.utils.data.DataLoader(PublicTestset, batch_size=opt.bs, shuffle=False, num_workers=1)
PrivateTestset = FER2013(split = 'PrivateTest', transform=transform_test)
PrivateTestloader = torch.utils.data.DataLoader(PrivateTestset, batch_size=opt.bs, shuffle=False, num_workers=1)



# Model
if opt.model == 'VGG19':
    net = VGG('VGG19')
elif opt.model  == 'Resnet18':
    net = ResNet18()

adversary = FGSMAttack(net,epsilon=0.3)

print('==> Building model..')
start_epoch = 1
if use_cuda:
    net.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    global Train_acc
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    if epoch > learning_rate_decay_start and learning_rate_decay_start >= 0:
        frac = (epoch - learning_rate_decay_start) // learning_rate_decay_every
        decay_factor = learning_rate_decay_rate ** frac
        current_lr = opt.lr * decay_factor
        utils.set_lr(optimizer, current_lr)  # set the decayed rate
    else:
        current_lr = opt.lr
    print('learning_rate: %s' % str(current_lr))
    tmp_loss = []
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        if opt.adv_train == 1:# and epoch > 5:
            y_pred = pred_batch(inputs,net)
            x_adv = adv_train(inputs,y_pred,net,criterion,adversary)
            x_adv_var = to_var(x_adv)
            loss_adv = criterion(net(x_adv_var),targets)
            loss = (loss+loss_adv)/2

        tmp_loss.append(loss)
        loss.backward()
        utils.clip_gradient(optimizer, 0.1)
        optimizer.step()
        train_loss += loss.data.item() #loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        utils.progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*float(correct)/float(total), correct, total))
    Loss_list.append(sum(tmp_loss)/len(tmp_loss))
    Train_acc = 100.*float(correct)/float(total)

def PublicTest(epoch):
    net.eval()
    PublicTest_loss = 0
    correct = 0
    total = 0
    param = {
        'test_batch_size': opt.bs,
        'epsilon': 0.3,
    }

    if opt.adv_test==1:
        PublicTest_adv_acc = attack_over_test_data(net,adversary,param,PublicTestloader)
        PublicTest_adv_acc_list.append(PublicTest_adv_acc)
        if PublicTest_adv_acc >= max(PublicTest_adv_acc_list):
            state = {
                'net': net.state_dict() if use_cuda else net,
                'acc': PublicTest_adv_acc,
                'epoch': epoch,
            }
            torch.save(state, os.path.join(path, 'PublicTest_adv_model.t7'))
    if opt.test==1:
        for batch_idx, (inputs, targets) in enumerate(PublicTestloader):
            # bs, ncrops, c, h, w = np.shape(inputs)
            # inputs = inputs.view(-1, c, h, w)
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            with torch.no_grad():
                inputs = Variable(inputs)
            targets = Variable(targets)
            outputs = net(inputs)
            # outputs_avg = outputs.view(bs, ncrops, -1).mean(1)  # avg over crops
            loss = criterion(outputs, targets)
            PublicTest_loss += loss.data.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            utils.progress_bar(batch_idx, len(PublicTestloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                           % (PublicTest_loss / (batch_idx + 1), 100.*float(correct)/float(total), correct, total))
        # Save checkpoint.
        PublicTest_acc = 100.*float(correct)/float(total)
        PublicTest_acc_list.append((PublicTest_acc))
        if PublicTest_acc >= max(PublicTest_acc_list):
            state = {
                'net': net.state_dict() if use_cuda else net,
                'acc': PublicTest_acc,
                'epoch': epoch,
            }
            torch.save(state, os.path.join(path, 'PublicTest_model.t7'))


def PrivateTest(epoch):
    net.eval()
    PrivateTest_loss = 0
    correct = 0
    total = 0
    param = {
        'test_batch_size': opt.bs,
        'epsilon': 0.3,
    }
    if opt.adv_test==1:
        PrivateTest_adv_acc = attack_over_test_data(net,adversary,param,PrivateTestloader)
        PrivateTest_adv_acc_list.append(PrivateTest_adv_acc)
        if PrivateTest_adv_acc >= max(PrivateTest_adv_acc_list):
            state = {
                'net': net.state_dict() if use_cuda else net,
                'acc': PrivateTest_adv_acc,
                'epoch': epoch,
            }
            torch.save(state, os.path.join(path, 'PrivateTest_adv_model.t7'))
    if opt.test == 1:
        for batch_idx, (inputs, targets) in enumerate(PrivateTestloader):
            # bs, ncrops, c, h, w = np.shape(inputs)
            # inputs = inputs.view(-1, c, h, w)
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            with torch.no_grad():
                inputs = Variable(inputs)
            targets = Variable(targets)
            #inputs, targets = Variable(inputs, volatile=True), Variable(targets)
            outputs = net(inputs)
            # outputs_avg = outputs.view(bs, ncrops, -1).mean(1)  # avg over crops
            loss = criterion(outputs, targets)
            PrivateTest_loss += loss.data.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            utils.progress_bar(batch_idx, len(PublicTestloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (PrivateTest_loss / (batch_idx + 1), 100.*float(correct)/float(total), correct, total))
        # Save checkpoint.
        PrivateTest_acc = 100.*float(correct)/float(total)
        PrivateTest_acc_list.append(PrivateTest_acc)
        if PrivateTest_acc >= max(PrivateTest_acc_list):
            state = {
                'net': net.state_dict() if use_cuda else net,
                'acc': PrivateTest_acc,
                'epoch': epoch,
            }
            torch.save(state, os.path.join(path, 'PrivateTest_model.t7'))


for epoch in range(start_epoch, total_epoch):
    train(epoch)
    PublicTest(epoch)
    PrivateTest(epoch)
    result = {
        'PublicTest_acc_list': PublicTest_acc_list,
        'PrivateTest_acc_list':PrivateTest_acc_list,
        'PublicTest_adv_acc_list':PublicTest_adv_acc_list,
        'PrivateTest_adv_acc_list':PrivateTest_adv_acc_list,
        'Loss_list':Loss_list
    }
    torch.save(result, os.path.join(path, 'Result.t7'))


print("best_PublicTest_acc: %0.3f" % max(PublicTest_acc_list))
print("best_PublicTest_adv_acc: %0.3f" % max(PublicTest_adv_acc_list))
print("best_PrivateTest_acc: %0.3f" % max(PrivateTest_acc_list))
print("best_PrivateTest_adv_acc: %0.3f" % max(PrivateTest_adv_acc_list))