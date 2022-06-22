# packages
import argparse
import logging
import time

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mid_params', default=100, type=int)
    parser.add_argument('--drop_p', default=0, type=float)
    parser.add_argument('--k', default=5, type=int)
    parser.add_argument('--n_epochs', default=200, type=int)
    parser.add_argument('--n_batch', default=16, type=int)
    parser.add_argument('--lr', default=.001, type=int)
    parser.add_argument('--wt_decay', default=0, type=float)
    return parser.parse_args()


def evaluation(model, cost, val_x, val_y):
    model = model.eval()

    val_x, val_y = val_x.float().cuda(), val_y.float().cuda()

    # forward
    with torch.no_grad():
        out = model(val_x)
        loss = cost(out, val_y.view(-1, 1)).item()

    model = model.train()
    return loss


def training(
    model, cost, opt,
    n_epochs, n_batch,
    train_x, train_y,
    val_x=None, val_y=None
    ):

    # create loader from tensor data
    data = TensorDataset(train_x, train_y)
    loader = DataLoader(data, n_batch, shuffle=True)

    # initialization
    model = model.train()
    train_loss_list = []
    val_loss_list = []

    for epoch in range(n_epochs):
        
        epoch_loss = epoch_n = 0
        # training
        for batch, (x, y) in enumerate(loader):
            x, y = x.float().cuda(), y.float().cuda()

            # forward
            out = model(x)
            loss = cost(out, y.view(-1, 1))

            # backward
            opt.zero_grad()
            loss.backward()
            opt.step()

            # results
            epoch_loss += loss.item() * y.size(0)
            epoch_n += y.size(0)

        # evaluation
        if val_x is not None:
            val_loss_list.append(
                evaluation(
                    model, cost, val_x, val_y
                )
            )

        # results
        train_loss_list.append(epoch_loss / epoch_n)
        
    return (train_loss_list, val_loss_list)


def get_fold_data(k, x, y):
    """ create iterables of test and train for k folds """

    # drop extra data
    n = len(x)
    m = n // k

    # random shuffle 
    i = torch.randperm(n)
    x, y = x[i], y[i]

    # initialization
    train_x_list, train_y_list = [], []
    val_x_list, val_y_list = [], []

    # create folds
    for j in range(k):

        # fold indices
        ind = range(j*m, j*m + m)
        not_ind = [h for h in range(n) if h not in ind]

        # append folds
        train_x_list.append(x[not_ind])
        train_y_list.append(y[not_ind])

        val_x_list.append(x[ind])
        val_y_list.append(y[ind])

    return (train_x_list, train_y_list, val_x_list, val_y_list)


def run_k_fold(
    k, 
    mid_params, drop_p, 
    x, y,
    n_epochs, n_batch, lr, wt_decay
    ):

    # logging 
    logger = logging.getLogger()
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',  # time stamps
        datefmt='%Y/%m/%d %H:%M:%S',  # time fmt
        level=logging.DEBUG,  # display to console
        handlers=[  # overwrite old logsLq
            logging.FileHandler('out_folds.log', 'a'), 
            logging.StreamHandler()
        ]
    )

    # import data
    (train_x_list, train_y_list, val_x_list, val_y_list) = get_fold_data(k, x, y)
    data = zip(train_x_list, train_y_list, val_x_list, val_y_list)

    # iterate over folds
    in_features = x.size(1)
    out_train = out_val = None

    for i, (train_x, train_y, val_x, val_y) in enumerate(data):
        start = time.time()

        # define model, loss, and optimizer
        model = nn.Sequential(
            nn.Dropout(drop_p),
            nn.Linear(in_features, mid_params),
            nn.Sigmoid(),
            nn.Dropout(drop_p),
            nn.Linear(mid_params, 1)
        ).cuda()
        if i == 0:
            logger.info(model)

        cost = nn.MSELoss().cuda() 
        opt = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=wt_decay
        )

        # results
        (train_loss_list, val_loss_list) = training(
            model, cost, opt, 
            n_epochs, n_batch, 
            train_x, train_y, 
            val_x, val_y
        )
        if out_train is None:
            out_train, out_val = np.array(train_loss_list), np.array(val_loss_list)
        else:
            out_train += np.array(train_loss_list)
            out_val += np.array(val_loss_list)

        end = time.time()
        print(f'Fold: {i + 1:d}, Time: {end - start:.1f}')

    # summarize avg fold
    out_train /= k
    out_val /= k

    out_train = out_train.tolist()
    out_val = out_val.tolist()

    val_min = min(out_val)
    val_min_i = out_val.index(val_min)
    train_i = out_train[val_min_i]

    logger.info(
        'Epoch: %d, Avg Train Loss: %.4f, Min Avg Val Loss: %.4f \n',
        val_min_i + 1, train_i, val_min
    )
    return (np.array(out_train), np.array(out_val))


def main():
    args = get_args()

    # import data
    train_x = torch.from_numpy(np.load('./data/train_x.npy'))
    train_y = torch.from_numpy(np.load('./data/train_y.npy'))
    in_features = train_x.size(1)

    # log transform
    train_y = torch.log(train_y)

    # k fold validation
    (train_out, val_out) = run_k_fold(
        args.k, 
        args.mid_params, args.drop_p,
        train_x, train_y, 
        args.n_epochs, args.n_batch, args.lr, args.wt_decay
    )

    # visualization
    logging.disable()
    plt.plot(range(1, 1 + args.n_epochs), train_out)
    plt.plot(range(1, 1 + args.n_epochs), val_out)
    plt.title(f'Avg Train/Validation Loss Over {args.k} Folds')
    plt.legend(labels=('train', 'validation'))
    plt.ylabel('MSELoss')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.savefig('loss_visual.png')

    return None


if __name__ == "__main__":
    main()
