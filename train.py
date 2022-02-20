from __future__ import print_function

import os
import random
import shutil
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
from config import args_wideresnet, args_preactresnet18
from utils import load_model, AverageMeter, accuracy
import logging
import sys

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


# Use CUDA

logger = get_logger('./exp.log')
# Use CUDA
use_cuda = torch.cuda.is_available()

seed = 11037
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, transform):
        images = np.load('./data.npy')
        labels = np.load('./label.npy')
        assert labels.min() >= 0
        assert images.dtype == np.uint8
        assert images.shape[0] <= 50000
        assert images.shape[1:] == (32, 32, 3)
        self.images = [Image.fromarray(x) for x in images]
        self.labels = labels / labels.sum(axis=1, keepdims=True) # normalize
        self.labels = self.labels.astype(np.float32)
        self.transform = transform
    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]
        image = self.transform(image)
        return image, label
    def __len__(self):
        return len(self.labels)


class MyvailDataset(torch.utils.data.Dataset):
    def __init__(self, transform):
        images = np.load('vail_data.npy')
        labels = np.load('vail_label.npy')
        assert labels.min() >= 0
        assert images.dtype == np.uint8
        assert images.shape[0] <= 50000
        assert images.shape[1:] == (32, 32, 3)
        self.images = [Image.fromarray(x) for x in images]
        self.labels = labels / labels.sum(axis=1, keepdims=True) # normalize
        self.labels = self.labels.astype(np.float32)
        self.transform = transform
    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]
        image = self.transform(image)
        return image, label
    def __len__(self):
        return len(self.labels)
    

def cross_entropy(outputs, smooth_labels):
    loss = torch.nn.KLDivLoss(reduction='batchmean')
    return loss(F.log_softmax(outputs, dim=1), smooth_labels)

def main():

    for arch in [ 'wideresnet','preactresnet18']:
        if arch == 'wideresnet':
            args = args_wideresnet
        else:
            args = args_preactresnet18
        print(arch)
        assert args['epochs'] <= 200
        if args['batch_size'] > 256:
            # force the batch_size to 256, and scaling the lr
            args['optimizer_hyperparameters']['lr'] *= 256/args['batch_size']
            args['batch_size'] = 256
        # Data
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_vail = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = MyDataset(transform=transform_train)
        vailset = MyvailDataset(transform=transform_vail)
        trainloader = data.DataLoader(trainset, batch_size=args['batch_size'], shuffle=True, num_workers=4)
        vailloader = data.DataLoader(vailset, batch_size=250, shuffle=False, num_workers=4)
        # Model

        model = load_model(arch)
        best_acc = 0  # best test accuracy

        optimizer = optim.__dict__[args['optimizer_name']](model.parameters(),
            **args['optimizer_hyperparameters'])
        if args['scheduler_name'] != None:
            scheduler = torch.optim.lr_scheduler.__dict__[args['scheduler_name']](optimizer,
            **args['scheduler_hyperparameters'])
        model = model.cuda()
        # Train and val
        logger.info('start training!')
        logger.info(args)
        print(args)
        for epoch in tqdm(range(args['epochs'])):

            train_loss, train_acc = train(trainloader, model, optimizer)
            
            #print('acc: {}'.format(train_acc))

            # save model
            best_acc = max(train_acc, best_acc)
            save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'acc': train_acc,
                    'best_acc': best_acc,
                    'optimizer' : optimizer.state_dict(),
                }, arch=arch)
            if args['scheduler_name'] != None:
                scheduler.step()
                
                
            vail_loss = 0
            vail_acc = 0
            """
            if epoch>-1:
                losses = AverageMeter()
                accs = AverageMeter()
                for (inputs, soft_labels) in vailloader:
                    inputs, soft_labels = inputs.cuda(), soft_labels.cuda()
                    outputs = model(inputs)
                    vail_targets = soft_labels.argmax(dim=1)
                    vail_loss = cross_entropy(outputs, soft_labels)
                    vail_acc = accuracy(outputs, vail_targets)
                    losses.update(vail_loss.item(), inputs.size(0))
                    accs.update(vail_acc[0].item(), inputs.size(0))
                vail_loss = losses.avg
                vail_acc = accs.avg
             """

            logger.info('Epoch:[{}/{}]\t loss={:.5f}\t acc={:.3f} vail_loss={:.5f}\t vail_acc={:.3f}'.format(epoch, args['epochs'], train_loss, train_acc,vail_loss,vail_acc))
        print('Best acc:')
        print(best_acc)
        logger.info('Best acc={:.3f}'.format(best_acc))
        logger.info('finish training!')
        if int(sys.argv[1])>0:
            shutil.move('./'+arch+'.pth.tar','./train_model/'+arch+'_'+sys.argv[1]+'.pth.tar')

def train(trainloader, model, optimizer):
    losses = AverageMeter()
    accs = AverageMeter()
    model.eval()
    # switch to train mode
    model.train()

    for (inputs, soft_labels) in trainloader:
        inputs, soft_labels = inputs.cuda(), soft_labels.cuda()
        targets = soft_labels.argmax(dim=1)
        outputs = model(inputs)
        loss = cross_entropy(outputs, soft_labels)
        acc = accuracy(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), inputs.size(0))
        accs.update(acc[0].item(), inputs.size(0))
    return losses.avg, accs.avg

def save_checkpoint(state, arch):
    filepath = os.path.join(arch + '.pth.tar')
    torch.save(state, filepath)

if __name__ == '__main__':
    print(sys.argv[1])
    main()
