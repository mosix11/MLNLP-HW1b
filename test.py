from src import DisCoTex
from src import Trainer
from src import SentClasLSTM, SentRegLSTM
from src import straified_baseline, simple_rnn_baseline
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import socket
import datetime

ds = DisCoTex(root_dir=Path('./data').absolute(), batch_size=1, tokenizer="bpe")

# simple_rnn_baseline(ds)
# ds = DisCoTex(tokenizer='bpe')
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

dl_train = ds.get_train_dataloader()
dl_val = ds.get_val_dataloader()
dl_test = ds.get_test_dataloader()

# print(get_avg_seq_len(dl_train))
print(ds.tokenizer.get_vocab_size())
# plot_data_dist(dl_train, dl_val, dl_test)





