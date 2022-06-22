# packages
import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
import random
import re
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


""" Model """ 
#
class VanillaRNN(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(VanillaRNN, self).__init__()

        # layers
        self.rnn = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=2, batch_first=False)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, inputs):
        
        out, hidden = self.rnn(inputs)
        out = self.linear(out)

        return out


""" Setup """
# file control
def get_args():
    parser = argparse.ArgumentParser()
    # control
    parser.add_argument('--f') 
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--debug', default=0, type=int)
    # training
    parser.add_argument('--num_iter', default=1000, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--hidden_size', default=256, type=int)
    parser.add_argument('--chunk_size', default=200, type=int)
    # optimizer
    parser.add_argument('--lr', default=.0025, type=float)
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


def perplexity(preds, y):

    # inputs of shape (n_seq, n_batch, n_char)
    cost = nn.CrossEntropyLoss().cuda()

    # undo one hot coding and put in (N, C)
    preds = preds.view(-1, preds.size()[-1])
    y = y.view(-1, y.size()[-1])

    return cost(preds, y)


""" Data manipulation """
# import and preprocess data
def load_data():
    
    """ Import data """
    # find start and stop 
    with open('gatsby.txt', mode='r', encoding="UTF-8") as f:
        lines = f.readlines()

    # start at first chapter
    start_ind = [i for i, line in enumerate(lines) if re.search('^\*\*\* START', line)]

    # stop at end of text
    stop_ind = [i for i, line in enumerate(lines) if re.search('^\*\*\* END', line)]

    # read file from start to stop
    with open('gatsby.txt', mode='r', encoding="UTF-8") as f:
        lines = f.readlines()[(start_ind[0] + 35):stop_ind[0]]

    """ Preprocessing """
    # remove extra whitespace and digits
    lines = [re.sub('[^A-Za-z ]', '', line) for line in lines]
    lines = [re.sub(' +', ' ', line) for line in lines]

    # remove extra linebreaks
    data = ' '.join(lines).lower()
    data = re.sub(' +', ' ', data)

    return data


# construct random batch
def get_batch(data, batch_size, chunk_size, num_char):
    
    # setup
    N = len(data)

    # initialization
    start = np.random.randint(low=0, high=N - chunk_size - 2)
    batch = data[start:(start + chunk_size + 1)]
    batch = F.one_hot(batch, num_char)
    batch = torch.unsqueeze(batch, dim=1)

    for i in range(batch_size - 1):

        start = np.random.randint(low=0, high=N - chunk_size - 2)
        temp = data[start:(start + chunk_size + 1)]
        temp = F.one_hot(temp, num_char)
        temp = torch.unsqueeze(temp, dim=1)
        batch = torch.cat([batch, temp], dim =1)

    outputs = batch[1:, ...]
    inputs = batch[:-1, ...]

    return (inputs.float().cuda(), outputs.float().cuda())


def tensor_to_char(data, int_to_char):

    data = torch.argmax(data[:, 0, :], dim=-1)
    data = torch.squeeze(data)

    data = data.cpu().numpy().tolist()
    out = [int_to_char[i] for i in data]
    out = ''.join(out)
    print(out)

    return None


# generate text
def generate(model, char_to_ind, ind_to_char, prefix='wh', seq_len=50):
    model = model.eval()
    
    # setup input
    ind_chars = torch.tensor([char_to_ind[char] for char in prefix])
    one_prefix = F.one_hot(ind_chars, len(char_to_ind)).float().cuda()
    one_prefix = torch.unsqueeze(one_prefix, dim=1)
    
    # generate string
    for i in range(seq_len - len(prefix)):

        # forward
        with torch.no_grad():
            char = model(one_prefix)

        char_probs = F.softmax(char[-1, :, :], dim=-1)
        sampled_char = torch.multinomial(char_probs, num_samples=1)
        sampled_char = F.one_hot(sampled_char, len(char_to_ind))

        # append to working string
        one_prefix = torch.cat(
            [one_prefix, sampled_char.view(1, 1, -1)],
            dim=0
        )

    one_prefix = torch.squeeze(one_prefix)  
    one_prefix = torch.argmax(one_prefix, dim=-1)

    # print text
    one_prefix = one_prefix.cpu().numpy().tolist()
    one_prefix = [ind_to_char[char] for char in one_prefix]
    one_prefix = ''.join(one_prefix)
    print(one_prefix)

    model = model.train()
    return None


""" Main function HERE """
def main():

    # hyperparmeters
    args = get_args()
    set_seeds(args.seed, args.debug)
    logger = get_log()

    """ Data """
    # tokenize
    text = load_data()
    ind_to_char = list(set(text))
    char_to_ind = dict([(char, i) for i, char in enumerate(ind_to_char)])
    char_size = len(char_to_ind)

    corpus_ind = [char_to_ind[char] for char in text]
    data = torch.LongTensor(corpus_ind)

    """ Model """
    # vanilla
    v_model = VanillaRNN(
        input_dim = char_size,
        hidden_dim = args.hidden_size,
        output_dim = char_size
    ).cuda()
    v_model = v_model.train()

    v_opt = optim.Adam(
        v_model.parameters(), lr=args.lr
    )

    """ Training """
    # initializations
    v_train_loss = []
    
    for i in range(args.num_iter):

        # batch setup
        start = time.time()
        inputs, outputs = get_batch(data, args.batch_size, args.chunk_size, char_size)

        """ Vanilla """
        # forward
        v_preds = v_model(inputs)
        if i % 5 == 0:
            tensor_to_char(inputs, ind_to_char)
            tensor_to_char(v_preds, ind_to_char)
        v_loss = perplexity(v_preds, outputs)

        # backward
        v_opt.zero_grad()
        v_loss.backward()
        nn.utils.clip_grad_norm_(v_model.parameters(), max_norm=1)
        v_opt.step()

        # results
        v_train_loss.append(v_loss.item())

        if args.debug:
            break

        # summary
        end = time.time()
        print(f'Iter: {i + 1}, Time: {end - start:.1f}, Vanilla Loss: {v_loss.item():.4f}')

        # generate samples
        if i % 5 == 0:
            print('\n---Vanilla---:')
            generate(v_model, char_to_ind, ind_to_char)

    """ Visualizations """
    # setup
    logging.disable()
    v_train_loss = np.array(v_train_loss)

    # loss
    plt.plot(range(1, len(v_train_loss) + 1), v_train_loss)
    plt.title('Average Train Loss Over Iterations')
    plt.ylabel('Cross Entropy Loss')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.legend(labels=['vanilla', 'gru'])
    plt.savefig('loss_visual.png')

            
if __name__ == "__main__":
    main()

