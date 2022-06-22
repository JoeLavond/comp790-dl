# packages
import argparse

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mid_params', default=100, type=int)
    parser.add_argument('--drop_p', default=0, type=float)
    parser.add_argument('--n_epochs', default=80, type=int)
    parser.add_argument('--n_batch', default=16, type=int)
    parser.add_argument('--lr', default=.001, type=int)
    parser.add_argument('--wt_decay', default=0, type=int)
    return parser.parse_args()


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


def main():
    args = get_args()

    # import data
    train_x = torch.from_numpy(np.load('./data/train_x.npy'))
    train_y = torch.from_numpy(np.load('./data/train_y.npy'))
    test_x = torch.from_numpy(np.load('./data/test_x.npy'))
    in_features = train_x.size(1)

    # log transform
    train_y = torch.log(train_y)

    # define model, loss, and optimizer
    model = nn.Sequential(
        nn.Dropout(args.drop_p),
        nn.Linear(in_features, args.mid_params),
        nn.Sigmoid(),
        nn.Dropout(args.drop_p),
        nn.Linear(args.mid_params, 1)
    ).cuda()

    cost = nn.MSELoss().cuda() 
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.wt_decay
    )

    # training
    model = model.train()
    (train_loss, _) = training(
        model, cost, opt,
        args.n_epochs, args.n_batch,
        train_x, train_y
    )

    # predictions
    model = model.eval()
    preds = model(test_x.float().cuda()).detach()
    preds = torch.exp(preds).cpu()
    preds = pd.Series(preds.reshape(1, -1)[0])

    # save for kaggle
    test_id = np.load('./data/test_id.npy')
    test_id = pd.Series(test_id)
    submission = pd.concat([test_id, preds], axis=1)
    submission.columns = ('Id', 'SalePrice')
    submission.to_csv('my_submission.csv', index=False)

    return None


if __name__ == "__main__":
    main()
