# packages
import argparse
import logging
import os
import sys
import time

import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torchvision import datasets, transforms as T

# source
from class_defs import *

# file control
def get_args():
    parser = argparse.ArgumentParser()
    # control
    parser.add_argument('--f') 
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--n_class', default=10, type=int)
    parser.add_argument('--n_folds', default=1, type=int)
    parser.add_argument('--debug', default=0, type=int)
    # training
    parser.add_argument('--n_epochs', default=200, type=int)
    parser.add_argument('--n_batch', default=128, type=int)
    # optimizer
    parser.add_argument('--lr', default=.2, type=float)
    parser.add_argument('--mom', default=.9, type=float)
    parser.add_argument('--wd', default=1e-4, type=float)
    return parser.parse_args()


# define and format logger
def get_log():

    logger = logging.getLogger()
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',  # time stamps
        datefmt='%Y/%m/%d %H:%M:%S',  # time fmt
        level=logging.DEBUG,  # display to console
        handlers=[  # overwrite old logsLq
            logging.FileHandler('./results.py', 'w+'), 
            logging.StreamHandler()
        ]
    )
    return logger


# make script reproducible
def set_seeds(seed, debug):

    # set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

    # turn off automatic debugs
    if not debug:
        torch.set_warn_always(False)
        torch.autograd.set_detect_anomaly(False)
        torch.autograd.profiler.profile(False)
        torch.autograd.profiler.emit_nvtx(False)
    
    return None


def evaluate(loader, model, cost):

    # initializations
    model = model.eval()
    val_loss = val_acc = val_n = 0

    # testing
    for batch, (images, labels) in enumerate(loader):
        images, labels = images.cuda(), labels.cuda()

        # forward
        with torch.no_grad():
            out = model(images)
            _, preds = out.max(dim=1)
            loss = cost(out, labels)

        # results
        val_loss += loss.item() * labels.size(0)
        val_acc += (labels == preds).sum().item()
        val_n += labels.size(0)

    # summarize
    val_loss /= val_n
    val_acc /= val_n
    model = model.train()
        
    return (val_loss, val_acc)


def training(loader, model, cost, opt, scheduler):

    # initializations
    train_loss = train_acc = train_n = 0

    # testing
    for batch, (images, labels) in enumerate(loader):
        images, labels = images.cuda(), labels.cuda()

        # forward
        out = model(images)
        _, preds = out.max(dim=1)
        loss = cost(out, labels)

        # backward
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        scheduler.step()

        # results
        train_loss += loss.item() * labels.size(0)
        train_acc += (labels == preds).sum().item()
        train_n += labels.size(0)

    # summarize
    train_loss /= train_n
    train_acc /= train_n
        
    return (train_loss, train_acc)


def main():

    # setup 
    args = get_args()
    logger = get_log()

    mu = (.4914, .4822, .4465)
    sd = (.2471, .2435, .2616)

    # transformations
    cifar_trans = T.Compose([
        T.ToTensor()
    ])

    # import data
    train_data = datasets.CIFAR10(
        root='./data/',
        train=True, 
        transform=cifar_trans, 
        download=True
    )
    assert len(train_data) % args.n_folds == 0

    if args.n_folds == 1:

        test_data = datasets.CIFAR10(
            root='./data/',
            train=False, 
            transform=cifar_trans,
            download=True
        )

        test_loader = DataLoader(
            test_data, 
            batch_size=512,
            shuffle=False,
            num_workers=1,
            pin_memory=True
        )

    else:

        folds_data = random_split(
            train_data, 
            lengths = [int(len(train_data) / args.n_folds) for i in range(args.n_folds)]
        )

    # initializations
    train_loss_out = np.zeros(shape=args.n_epochs)
    train_acc_out = np.zeros(shape=args.n_epochs)
    val_loss_out = np.zeros(shape=args.n_epochs)
    val_acc_out = np.zeros(shape=args.n_epochs)

    # perform k-fold cross validation
    for fold_i in range(args.n_folds):

        if args.n_folds == 1:

            train_loader = DataLoader(
                test_data, 
                batch_size=args.n_batch,
                shuffle=True,
                num_workers=1,
                pin_memory=True
            )

        else:

            train_loader = DataLoader(
                ConcatDataset([fold for i, fold in enumerate(folds_data) if i != fold_i]),
                batch_size=args.n_batch,
                shuffle=True,
                num_workers=1, 
                pin_memory=True
            )

            val_loader = DataLoader(
                folds_data[fold_i],
                batch_size=512,
                shuffle=False, 
                num_workers=1, 
                pin_memory=True
            )

        # define model, cost, and optimizer
        model = nn.Sequential(
            StdLayer(mu, sd),
            PreActResNet18()
        ).cuda()
        model = model.train()

        cost = nn.CrossEntropyLoss().cuda()

        opt = optim.SGD(
            model.parameters(),
            lr=args.lr, momentum=args.mom, weight_decay=args.wd
        )

        scheduler = optim.lr_scheduler.OneCycleLR(
            opt, max_lr=args.lr, 
            epochs=args.n_epochs, steps_per_epoch=len(train_loader)
        )

        # initializations
        fold_train_loss = []
        fold_train_acc = []
        fold_val_loss = []
        fold_val_acc = []

        # main loop
        for epoch in range(args.n_epochs):

            # training 
            (train_loss, train_acc) = training(train_loader, model, cost, opt, scheduler)
            fold_train_loss.append(train_loss)
            fold_train_acc.append(train_acc)

            if args.n_folds == 1:

                # results
                logger.info(
                    'Epoch %d, Train Loss: %.4f, Train Acc: %.4f',
                    epoch + 1, train_loss, train_acc
                )

            else:

                # validation
                (val_loss, val_acc) = evaluate(val_loader, model, cost)
                fold_val_loss.append(val_loss)
                fold_val_acc.append(val_acc)

                # results
                logger.info(
                    'Epoch %d, Train Loss: %.4f, Train Acc: %.4f, Val Loss: %.4f, Val Acc: %.4f', 
                    epoch + 1, train_loss, train_acc, val_loss, val_acc
                )

            if args.debug:
                break

        # summary
        train_loss_out += np.array(fold_train_loss)
        train_acc_out += np.array(fold_train_acc)
        if args.n_folds != 1:
            val_loss_out += np.array(fold_val_loss)
            val_acc_out += np.array(fold_val_acc)

    # evaluate final model
    train_loss_out /= args.n_folds
    train_acc_out /= args.n_folds

    if args.n_folds == 1:
        evaluate(test_loader, model, cost)
    else:
        val_loss_out /= args.n_folds
        val_acc_out /= args.n_folds

        val_acc_cp = val_acc_out.tolist()
        max_val_acc = max(val_acc_cp)
        max_val_i = val_acc_cp.index(max_val_acc)
        logger.info('Max Val Acc: %.4f, Epoch: %d', max_val_acc, max_val_i + 1)

    # loss visual
    logging.disable()
    plt.plot(range(1, args.n_epochs + 1), train_loss_out)
    plt.title('Average Train Loss Over Folds')
    plt.ylabel('Cross Entropy Loss')
    plt.yscale('log')
    plt.xlabel('Epoch')
    if args.n_folds == 1:
        plt.plot(range(1, args.n_epochs + 1), val_loss_out)
        plt.title('Average Train And Validatation Loss Over Folds')
    plt.savefig('loss_visual.png')

    # acc visual
    plt.plot(range(1, args.n_epochs + 1), train_acc_out)
    plt.title('Average Train Acc Over Folds')
    plt.ylabel('Cross Entropy Acc')
    plt.yscale('log')
    plt.xlabel('Epoch')
    if args.n_folds == 1:
        plt.plot(range(1, args.n_epochs + 1), val_acc_out)
        plt.title('Average Train And Validatation Acc Over Folds')
    plt.savefig('acc_visual.png')



if __name__ == "__main__":
    main()
