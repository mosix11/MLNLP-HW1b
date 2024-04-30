import torch
import os
import sys
import argparse
from pathlib import Path
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

import socket
import datetime

from src import DisCoTex
from src import Trainer
from src import SentClasLSTM, SentRegLSTM, SentRegAttLSTM, HierarchicalSentRegLSTM
from src import straified_baseline, simple_rnn_baseline

def get_label_dist(dl):
    i = 0
    label_freqs = {0: 0, 1: 0, 2: 0, 3: 0}
    for batch in dl:
        i += 1
        paded, lens, labels = batch
        for l in labels:
            label_freqs[int(l)] = label_freqs[int(l)] + 1

    training_data_count = label_freqs[0] + label_freqs[1] + label_freqs[2] + label_freqs[3]
    for l in label_freqs:
        label_freqs[int(l)] = label_freqs[int(l)] / training_data_count
        
    return label_freqs

def plot_data_dist(dl_train, dl_val, dl_test):
    train_dist = get_label_dist(dl_train)
    val_dist = get_label_dist(dl_val)
    test_dist = get_label_dist(dl_test)

    socres = train_dist.keys()

    x = np.arange(1, len(socres)+1)  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width, train_dist.values(), width, label='Train')
    rects2 = ax.bar(x , val_dist.values(), width, label='Val')
    rects3 = ax.bar(x + width, test_dist.values(), width, label='Test')

    ax.set_ylabel('Probability')
    ax.set_title('Label Distribution in Dataset Splits')
    ax.set_xticks(x)
    ax.set_xticklabels(x)
    ax.legend()

    fig.tight_layout()
    fig.set_dpi(150)

    sum_writer = SummaryWriter(Path('./outputs/tensorboard/').joinpath(socket.gethostname()+'-'+str(datetime.datetime.now())).absolute())

    sum_writer.add_figure('Fig1', fig)
    sum_writer.close()
    
def get_avg_seq_len(dl):
    num_sample = 0
    cum_len = 0
    
    for batch in dl:
        paded, lens, labels = batch
        for seq in paded:
            cum_len += seq.shape[0]
            num_sample += 1
            
    return cum_len / num_sample 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", help="whether to use base line or not. Set bs1 to run the simple baseline and bs2 for the rnn baseline", type=str, default=None)
    parser.add_argument("-m", "--model", help="specify model type between \{SC, SR, SRA, HSR\}", type=str, default="SC")
    parser.add_argument("-t", "--tokenizer", help="specify tokenizer type between \{word, bpe\}", type=str, default="word")
    parser.add_argument("-e", "--epoch", help="number of epochs", type=int, default=20)
    parser.add_argument("-b", "--batch", help="batch size", type=int, default=64)
    parser.add_argument("-g", "--gpu", help="whether to use GPU for training or not", action="store_true")
    parser.add_argument("-s", "--save", help="where to save trained model", type=str, default="./weights/model.pt")
    parser.add_argument("-r", "--lr", help="learning rate", type=float, default=1e-4)
    parser.add_argument("--lrs", help="use lr scheduler", action="store_true")
    args = parser.parse_args()
    
    if args.base == 'bs1':
        ds = DisCoTex(root_dir=Path('./data').absolute(), batch_size=32, tokenizer="word")
        straified_baseline(ds)
        exit()
    elif args.base == 'bs2':
        ds = DisCoTex(root_dir=Path('./data').absolute(), batch_size=64, tokenizer="word")
        simple_rnn_baseline(ds)
        exit()
        
    
    dataset = DisCoTex(root_dir=Path('./data'), outputs_dir=Path('./outputs'), batch_size=args.batch, tokenizer=args.tokenizer)
    if args.model == "SR":
        model = SentRegLSTM(
            vocab_size=dataset.tokenizer.get_vocab_size(),
            embed_dim=128,
            hidden_size=128,
            num_layers=2,
            bidirectional=True,
            padd_index=dataset.tokenizer.get_pad_idx(),
            dropout=0.3
        )
    elif args.model == "SC":
        model = SentClasLSTM(
            vocab_size=dataset.tokenizer.get_vocab_size(),
            embed_dim=128,
            hidden_size=128,
            num_layers=3,
            bidirectional=True,
            padd_index=dataset.tokenizer.get_pad_idx(),
            dropout=0.3
        )
    elif args.model == "SRA":
        model = SentRegAttLSTM(
            vocab_size=dataset.tokenizer.get_vocab_size(),
            embed_dim=96,
            hidden_size=64,
            num_layers=3,
            bidirectional=True,
            padd_index=dataset.tokenizer.get_pad_idx(),
            dropout=0.3
        )
    elif args.model == "HSR":
        model = HierarchicalSentRegLSTM(
            vocab_size=dataset.tokenizer.get_vocab_size(),
            embed_dim=128,
            hidden_size=64,
            num_layers=3,
            bidirectional=True,
            padd_index=dataset.tokenizer.get_pad_idx(),
            dropout=0.5,
            loss_fn="MSE"
        )
    
    
    trainer = Trainer(max_epochs=args.epoch, lr=args.lr, optimizer_type="adam", use_lr_schduler=args.lrs, run_on_gpu=True)
    trainer.fit(model, dataset)
    
    if not os.path.exists(Path(args.save).parent):
        os.mkdir(Path(args.save).parent)
    torch.save(model, Path(args.save))
    
    # ds = DisCoTex(root_dir=Path('./data').absolute(), batch_size=1, tokenizer="bpe")
    # dl_train = ds.get_train_dataloader()
    # dl_val = ds.get_val_dataloader()
    # dl_test = ds.get_test_dataloader()

    # print(get_avg_seq_len(dl_train))
    # print(ds.tokenizer.get_vocab_size())
    # plot_data_dist(dl_train, dl_val, dl_test)