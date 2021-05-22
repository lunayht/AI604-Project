from __future__ import print_function
import os
import argparse
import torchvision

import torch.optim as optim
import math
import numpy as np
import torch
from torch import nn, einsum
import torch.nn.functional as F
from pytorch_vit import *

import torchvision.transforms as transforms

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
criterion = nn.CrossEntropyLoss()


def train(epochs, epoch, model, train_loader, optimizer):
    model.train()
    global history_score
    avg_loss = 0.
    train_acc = 0.
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        avg_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        train_acc += (predicted == target).sum().item()
        # epoch_loss += loss / len(train_loader
        # pred = output.data.max(1, keepdim=True)[1]
        # train_acc += pred.eq(target.data.view_as(pred)).cpu().sum()
        loss.backward()

        optimizer.step()
        # if batch_idx % 100 == 0:
        #     print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),
        #                                                          len(train_loader.dataset), loss.item()))
    epoch_loss = avg_loss / total
    epoch_acc = 100.0 * train_acc / total
    print(f'Epoch: {epoch + 1}/{epochs} Loss: {epoch_loss:.4f} '
          f' train acc:{epoch_acc:.2f}')
    history_score[epoch][0] = avg_loss / len(train_loader)
    history_score[epoch][1] = train_acc / float(len(train_loader))


def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        output = model(data)
        test_loss += criterion(output, target).item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        # pred = output.data.max(1, keepdim=True)[1]
        # correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(test_loss, correct,
                                                                                 len(test_loader.dataset),
                                                                                 100. * correct / total))
    return correct / total


class CustomPad(object):
    # making the image size divisible py patch size
    def __init__(self, patch_size, background=(0, 0, 0)):
        self.fill_with = background
        self.right = 0
        self.left = 0
        self.top = 0
        self.bott = 0
        self.p1 = patch_size[0]
        self.p2 = patch_size[1]

    def __call__(self, image):
        width, height = image.size
        if (width % self.p1) != 0:
            wdelta = self.p1 - (width % self.p1)
            if (wdelta % 2) != 0:
                self.left = wdelta // 2 + 1
                self.right = wdelta // 2
            else:
                self.left = wdelta // 2
                self.right = wdelta // 2
        if (height % self.p2) != 0:
            hdelta = self.p2 - (height % self.p2)
            if (hdelta % 2) != 0:
                self.top = hdelta // 2 + 1
                self.bott = hdelta // 2
            else:
                self.top = hdelta // 2
                self.bott = hdelta // 2

        padding = (self.left, self.top, self.right, self.bott)
        transform = transforms.Pad(padding, fill=self.fill_with)
        return transform(image)


if __name__ == '__main__':

    # Training settings
    parser = argparse.ArgumentParser(description='traing  architecture setting')
    parser.add_argument('--data', type=str, default='./data/data_40X', help='path to dataset')
    parser.add_argument('--num_classes', type=int, default=2, help='number of classes')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N'
                        , help='batch size (default:256)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 160)')

    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', '--wd', default=0.0001, type=float,
                        metavar='W', help='weight decay (default: 1e-3)')
    # parser.add_argument('--seed', type=int, default=42, metavar='S',
    #                     help='random seed (default: 1)')
    parser.add_argument('--save', default='./checkpoint', type=str, metavar='PATH',
                        help='path to save prune model (default: current directory)')
    parser.add_argument('--patch_size', default=(128, 128), type=tuple, help='patch size to use')
    parser.add_argument('--arch_type', type=str, default='vit')

    # parser.add_argument('--threshold', type=float, default = 0.15)
    args = parser.parse_args()
    # print(args)
    seed = 42
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    patch_size = args.patch_size
    # print(patch_size)
    batch_size = args.batch_size
    im_size = (460, 700)  # We'll resize input images to this size
    # image_size = (460, 700)
    # patch_size = (64, 64)  # Size of the patches to be extract from the input images
    num_patches = math.ceil((im_size[0] / patch_size[0])) * math.ceil((im_size[1] / patch_size[1]))
    # transforms.Normalize((0.80, 0.65, 0.77), (0.11, 0.15, 0.11))])
    image_size = (
        patch_size[0] * math.ceil((im_size[0] / patch_size[0])),
        patch_size[1] * math.ceil((im_size[1] / patch_size[1])))

    transform = transforms.Compose([transforms.Resize((460, 700)),
                                    # transforms.RandomResizedCrop((460, 700)),
                                    CustomPad(patch_size),
                                    #                                 transforms.RandomResizedCrop((460,700)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    test_transform = transforms.Compose([transforms.Resize((460, 700)),
                                         CustomPad(patch_size),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    if args.num_classes == 2:
        print('-----------------loading data ----------')
        trainset = torchvision.datasets.ImageFolder(root='./data/data_100X/train/',
                                                    transform=transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                   shuffle=True, num_workers=4)

        testset = torchvision.datasets.ImageFolder(root='./data/data_100X/test/',
                                                   transform=test_transform)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                  shuffle=False, num_workers=4)
    else:
        print('-----------------loading data ----------')
        trainset = torchvision.datasets.ImageFolder(root='./data/data_40X/Mtrain/',
                                                    transform=transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                   shuffle=True, num_workers=4)

        testset = torchvision.datasets.ImageFolder(root='./data/data_40X/Mtest/',
                                                   transform=test_transform)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                  shuffle=False, num_workers=4)
    print('-----------------defining the network model ----------')
    model = ViT(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=2,
        dim=1024,
        depth=6,
        heads=16,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1
    ).cuda()
    print('-----------------model paramters assigments----------')
    model = nn.DataParallel(model)
    model = model.cuda()

    epochs = args.epochs
    # lr = 3e-5
    lr = args.lr
    # gamma = 0.7

    # num_patches = math.ceil((im_size[0] / patch_size[0])) * math.ceil((im_size[1] / patch_size[1]))

    # image_size = (
    # patch_size[0] * math.ceil((im_size[0] / patch_size[0])), patch_size[1] * math.ceil((im_size[1] / patch_size[1])))

    # loss function
    # criterion = nn.CrossEntropyLoss()
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # scheduler
    scheduler = StepLR(optimizer, step_size=50, gamma=0.1)
    #
    #
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # optimizer = optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay, amsgrad=False)

    if not os.path.exists(args.save):
        os.mkdir(args.save)

    # model_dir = os.path.join(args.save,'model_'+ str(args.epochs)+'_'+ str(args.arch_type)+'.pth')

    history_score = np.zeros((args.epochs + 1, 3))
    model_dir = os.path.join(args.save, 'model_' + str(args.arch_type) + '.pth')
    print('-----------------starting training ----------')

    best_prec1 = 0.
    for epoch in range(args.epochs):

        train(args.epochs, epoch, model, train_loader, optimizer)
        prec1 = test(model, test_loader)
        history_score[epoch][2] = prec1

        if prec1 > best_prec1:
            best_prec1 = prec1
            torch.save(model.module.state_dict(), model_dir)
        scheduler.step()

    print("Best accuracy: " + str(best_prec1))
    print('-----------------Finished ----------')

