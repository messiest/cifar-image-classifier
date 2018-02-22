#! /usr/bin/env/ python3.6
import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms

from cifar import data
from models.googlenet import GoogLeNet


def main(net, resume=False):
    print('Running...')
    net = net()
    net.zero_grad()
    print(net)
    cuda = torch.cuda.is_available()  # Hopefully you can you use GPUs
    if cuda:  # set up GPU parallelization
        net.cuda()
        net = torch.nn.DataParallel(
            net,
            device_ids=range(torch.cuda.device_count())
        )
        cudnn.benchmark = True

    train_loader, test_loader = data.get()

    criterion = nn.CrossEntropyLoss()  # loss function
    optimizer = optim.Adam(net.parameters())  # optimization algorithm

    best_acc = 0

    def training(epoch):
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for i, (inputs, targets) in enumerate(train_loader):
            if cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()  # back propagation
            optimizer.step()  # update algorithm

            train_loss += loss.data[0]
            # print("OUTPUTS:", outputs, type(outputs))
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            print("{} - {}%".format(i, 100.*correct/total))

    def testing(epoch):
        nonlocal best_acc

        net.eval()
        test_loss = 0
        correct = 0
        total = 0

        for i, (inputs, targets) in enumerate(test_loader):
            if cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.data[0]
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

        acc = 100. * correct/total
        if acc > best_acc:
            print('Saving...')
            state = {
                'net': net.module if cuda else net,
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt.t7')  # ?
            best_acc = acc

    for epoch in range(200):
        print("Training...")
        training(epoch)
        print("Testing...")
        testing(epoch)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        resume = sys.argv[1]
        if resume:
            assert os.path.isdir('checkpoint'), \
                'Error: no checkpoint directory found!'
            checkpoint = torch.load('./checkpoint/ckpt.t7')
            net = checkpoint['net']
            main(net)
    else:
        main(GoogLeNet)
