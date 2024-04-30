import torch
import os
import sys
import argparse
from pathlib import Path

from src import DisCoTex
from src import Trainer
from src import SentClasLSTM, SentRegLSTM, SentRegAttLSTM, HierarchicalSentRegLSTM
from src import straified_baseline, simple_rnn_baseline


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
            embed_dim=128,
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
    
