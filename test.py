from src import DisCoTex
from src import Trainer
from src import SentClasLSTM, SentRegLSTM
from src import straified_baseline
import torch
from pathlib import Path

ds = DisCoTex(root=Path('./data'), batch_size=32, tokenizer="bpe")

straified_baseline(ds)
# ds = DisCoTex(tokenizer='bpe')

# dl = ds.get_train_dataloader()
# i = 0
# label_freqs = {0: 0, 1: 0, 2: 0, 3: 0}
# for batch in dl:
#     i += 1
#     paded, lens, labels = batch
#     for l in labels:
#         label_freqs[int(l)] = label_freqs[int(l)] + 1
    # print(type(lens), lens.shape)
    # print(paded.shape)
    # print(type(labels), labels.shape)
    # print(ds.tokenizer.decode(paded[0].tolist()))
    # exit()
    
# print(label_freqs)
