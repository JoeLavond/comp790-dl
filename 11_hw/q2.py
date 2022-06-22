""" Imports """
# basic
import argparse
import logging
import os
import time

# numerical
import matplotlib.pyplot as plt
import numpy as np
import random

# pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# text
import gensim.downloader as api
from transformers import DistilBertTokenizer, DistilBertModel

""" Class definitions """
class CustomDataset(Dataset):

    def __init__(self, x=None, y=None, obj=None, trans=None, init=1):
        self.trans = trans

        # model with average sentence vector
        if obj:
            self.x = obj['input_ids']
            self.y = obj['labels']
        else:
            self.x, self.y = x, y

    def __getitem__(self, index):
        x, y = self.x[index], self.y[index]

        if self.trans is not None:
            x = self.trans(x)

        return x, y

    def __len__(self):
        return len(self.y)

    def split_k(self, k):

        # setup
        n = len(self)
        m = n // k
        assert n % k == 0

        # shuffle
        ind = torch.randperm(n)

        # create folds
        output = list()
        for fold_i in range(k):
            temp_ind = torch.randperm(n)[fold_i*m:(fold_i*m + m)]
            output.append(
                CustomDataset(
                    self.x.clone()[temp_ind], 
                    self.y.clone()[temp_ind],
                    trans=self.trans, 
                    init=0
                )
            )
        return output


# concatenate list of custom datasets to one
def CustomConcat(data_list):

    # start output
    out_x = data_list[0].x
    out_y = data_list[0].y
    out_trans = data_list[0].trans

    # iteratively append to new dataset
    for i, data in enumerate(data_list):

        if i == 0:
            continue
        else:

            # append to new
            out_x = torch.cat((out_x, data.x))
            out_y = torch.cat((out_y, data.y))

    output = CustomDataset(out_x, out_y, trans=out_trans, init=0)
    return output


""" Helper functions """
# import data as tuples of strs and ints
def load_sst2_data(filename):
    with open(filename) as f:
        data = [(l[2:].strip(), int(l[0])) for l in f.readlines()]
    return tuple(zip(*data))


# allows list comprehension of sentence
def vec_or_none(word, word_vec_obj):
    try:
        return word_vec_obj[word]
    except KeyError:
        return None

# take a single strs of data and return the word vecs for each word in sentence (word_i, word_vec_i)
def sentence_vecs(sentence, word_vec_obj):

    # split on white space
    word_list = sentence.split()

    # convert words to vectors
    word_vec_list = [vec_or_none(word, word_vec_obj) for word in word_list]
    word_vec_list = [word_vec for word_vec in word_vec_list if word_vec is not None]

    # return array
    word_vec_arr = np.array(word_vec_list)
    return word_vec_arr


""" Setup functions """
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f')
    parser.add_argument('--data', default='../data', type=str)
    parser.add_argument('--seed', default=0, type=int)
    # control
    parser.add_argument('--n_folds', default=1, type=int)
    parser.add_argument('--debug', default=0, type=int)
    # training
    parser.add_argument('--fine_tune', default=0, type=int)
    parser.add_argument('--n_epochs', default=20, type=int)
    parser.add_argument('--n_batch', default=8, type=int)
    parser.add_argument('--step', default=1, type=int)
    # optimizer
    parser.add_argument('--lr', default=0.05, type=float)
    parser.add_argument('--mom', default=.9, type=float)
    parser.add_argument('--wd', default=5e-4, type=float)
    return parser.parse_args()


# define and format logger
def get_log(path='.'):

    logger = logging.getLogger()
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',  # time stamps
        datefmt='%Y/%m/%d %H:%M:%S',  # time fmt
        level=logging.DEBUG,  # display to console
        handlers=[  # overwrite old logsLq
            logging.FileHandler(os.path.join(path, 'out_train.log'), 'w+'), 
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

    return None


""" Training and testing """
def evaluate(loader, pt1, pt2, cost, fine_tune, **kwargs):

    # initializations
    pt1, pt2 = pt1.eval(), pt2.eval()
    test_loss = test_acc = test_n = 0

    # testing
    test_start = time.time()
    for batch, (x, y) in enumerate(loader):
        x, y = x.cuda(), y.cuda()

        # forward
        with torch.no_grad():
            pt1_out = pt1(x)
            pt1_out = pt1_out['last_hidden_state'][:, 0, :]
            pt2_out = torch.squeeze(pt2(pt1_out))
            loss = cost(pt2_out, y)

        # results
        test_loss += loss.item() * y.size(0)
        test_acc += (pt2_out.round() == y).sum().item()
        test_n += y.size(0)

    # summarize
    test_loss /= test_n
    test_acc /= test_n
    test_end = time.time()

    if fine_tune:
        pt1 = pt1.train()
    pt2 = pt2.train()

    return (test_end - test_start, test_loss, test_acc)


def training(loader, pt1, pt2, cost, opt, debug, fine_tune, step, scheduler=None, **kwargs):

    # initializations
    train_loss = train_acc = train_n = 0

    # training
    train_start = time.time()
    for batch, (x, y) in enumerate(loader):
        x, y = x.cuda(), y.cuda()

        # forward
        if not fine_tune:
            with torch.no_grad():
                pt1_out = pt1(x)
        else:
            pt1_out = pt1(x)

        pt1_out = pt1_out['last_hidden_state'][:, 0, :]
        pt2_out = torch.squeeze(pt2(pt1_out))
        loss = cost(pt2_out, y)

        # backward
        loss.backward()
        if (batch + 1) % step == 0:
            opt.step()
            opt.zero_grad(set_to_none=True)
            if scheduler: 
                scheduler.step()

        # results
        train_loss += loss.item() * y.size(0)
        train_acc += (pt2_out.round() == y).sum().item()
        train_n += y.size(0)

        if debug:
            break

    # summarize
    train_end = time.time()
    train_loss /= train_n
    train_acc /= train_n

    opt.zero_grad(set_to_none=True)
    return (train_end - train_start, train_loss, train_acc)


""" Main function HERE """
def main():

    # setup
    args = get_args()

    args.k_fold = (args.n_folds > 1)
    args.path = './k_fold' if args.k_fold else './output'

    set_seeds(args.seed, args.debug)
    logger = get_log(args.path)
    logger.info(args)

    # import data
    train_sentences, train_labels = load_sst2_data(
        os.path.join(args.data, "stsa.binary.train")
    )
    test_sentences, test_labels = load_sst2_data(
        os.path.join(args.data, "stsa.binary.test")
    )

    # tokenize sentences
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    train_data = tokenizer(list(train_sentences), padding='max_length', return_tensors='pt')
    train_data['labels'] = torch.from_numpy(np.array(train_labels)).float()
    
    test_data = tokenizer(list(test_sentences), padding='max_length', return_tensors='pt')
    test_data['labels'] = torch.from_numpy(np.array(test_labels)).float()

    train_data = CustomDataset(obj=train_data)

    # import test or split train
    if not args.k_fold:

        test_data = CustomDataset(obj=test_data)
        test_loader = DataLoader(
            test_data, 
            batch_size=512, 
            shuffle=False, 
            num_workers=1, 
            pin_memory=True
        )

    else:

        folds_data = train_data.split_k(args.n_folds)
        val_out_loss = np.zeros(shape=args.n_epochs)
        val_out_acc = np.zeros(shape=args.n_epochs)

    # necessary inits
    cost = nn.BCELoss().cuda()
    train_out_loss = np.zeros(shape=args.n_epochs)
    train_out_acc = np.zeros(shape=args.n_epochs)

    for fold_i in range(args.n_folds):

        if not args.k_fold:

            # create training loader from full train
            train_loader = DataLoader(
                train_data, 
                batch_size=args.n_batch,
                shuffle=True,
                num_workers=1,
                pin_memory=True
            )

        else:
            logger.info('\n\n---Fold %d---\n', fold_i + 1)

            """ Combine training folds """
            to_train = CustomConcat(
                [fold for i, fold in enumerate(folds_data) if i != fold_i]
            )

            # create training loader
            train_loader = DataLoader(
                to_train, 
                batch_size=args.n_batch,
                shuffle=True,
                num_workers=1,
                pin_memory=True
            )

            """ For the validation set """
            # no transformations for validation data
            val_data = CustomDataset(
                folds_data[fold_i].x, folds_data[fold_i].y, init=0 
            )
    
            # create validation loaders
            val_loader = DataLoader(
                val_data, 
                batch_size=512,
                shuffle=True,
                num_workers=1,
                pin_memory=True
            )

            # initializations
            val_temp_loss = []
            val_temp_acc = []

        """ Define model and optimizer """
        pt1 = DistilBertModel.from_pretrained('distilbert-base-uncased').cuda()
        pt1 = pt1.eval()
        if args.fine_tune:
            pt1 = pt1.train()

        pt2 = nn.Sequential(
            nn.Linear(in_features=768, out_features=1),
            nn.Sigmoid()
        ).cuda()
        pt2 = pt2.train()

        opt = optim.AdamW(
            pt2.parameters() if not args.fine_tune else list(pt1.parameters()) + list(pt2.parameters()), 
            lr=args.lr#, momentum=args.mom, weight_decay=args.wd
        )

        scheduler = optim.lr_scheduler.OneCycleLR(
            opt, max_lr=args.lr, 
            epochs=args.n_epochs, steps_per_epoch=len(train_loader)
        )

        """ Main training and validation loop """
        # initializations
        train_temp_loss = []
        train_temp_acc = []
        
        for epoch in range(args.n_epochs):
            
            # training
            (train_time, train_loss, train_acc) = training(
                train_loader, pt1, pt2, cost, opt, **vars(args)#, scheduler=scheduler
            )
            train_temp_loss.append(train_loss)
            train_temp_acc.append(train_acc)

            # summarize 
            logger.info(
                'TRAINING - Epoch: %d, Time: %.1f, Loss: %.4f, Acc: %.4f',
                epoch + 1, train_time, train_loss, train_acc
            )

            if args.k_fold:

                # validation
                (val_time, val_loss, val_acc) = evaluate(
                        val_loader, pt1, pt2, cost, **vars(args)
                )
                val_temp_loss.append(val_loss)
                val_temp_acc.append(val_acc)

                logger.info(
                    'VALIDATION - Epoch: %d, Time: %.1f, Loss: %.4f, Acc: %.4f',
                    epoch + 1, val_time, val_loss, val_acc
                )

            #if args.debug:
            #    break

        # results
        train_out_loss += np.array(train_temp_loss)
        train_out_acc += np.array(train_temp_acc)

        if args.k_fold:
            val_out_loss += np.array(val_temp_loss)
            val_out_acc += np.array(val_temp_acc)


    # final evaluations
    if not args.k_fold:

        """ Testing """
        # testing
        (test_time, test_loss, test_acc) = evaluate(
            test_loader, pt1, pt2, cost, **vars(args)
        )

        logger.info(
            'TESTING - Epoch: %d, Time: %.1f, Loss: %.4f, Acc: %.4f',
            epoch + 1, test_time, test_loss, test_acc
        )

    else:

        # average performance across folds
        train_out_loss /= args.n_folds
        train_out_acc /= args.n_folds
        val_out_loss /= args.n_folds
        val_out_acc /= args.n_folds

    """ Visualizations """
    # loss visual
    logging.disable()
    plt.figure()
    plt.plot(range(1, args.n_epochs + 1), train_out_loss)
    plt.title('Average Train Loss Over Folds')
    plt.ylabel('Cross Entropy Loss')
    plt.yscale('log')
    plt.xlabel('Epoch')
    if args.k_fold:
        plt.plot(range(1, args.n_epochs + 1), val_out_loss)
        plt.legend(labels=['train', 'validation'])
        plt.title('Average Train And Validatation Losses Over Folds')
    plt.savefig(
        os.path.join(args.path, 'loss_visual.png')
    )

    # acc visual
    logging.disable()
    plt.figure()
    plt.plot(range(1, args.n_epochs + 1), train_out_acc)
    plt.title('Average Train Acc Over Folds')
    plt.ylabel('Acc')
    plt.xlabel('Epoch')
    if args.k_fold:
        plt.plot(range(1, args.n_epochs + 1), val_out_acc)
        plt.legend(labels=['train', 'validation'])
        plt.title('Average Train And Validatation Acces Over Folds')
    plt.savefig(
        os.path.join(args.path, 'acc_visual.png')
    )


if __name__ == "__main__":
    main()

