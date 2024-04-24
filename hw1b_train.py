import torch
import os
import sys
import argparse
from pathlib import Path

from src import DisCoTex
from src import Trainer
from src import SentClasLSTM, SentRegLSTM



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="specify model type between \{SC, SR\}", type=str, default="SR")
    parser.add_argument("-t", "--tokenizer", help="specify tokenizer type between \{word, bpe\}", type=str, default="word")
    parser.add_argument("-e", "--epoch", help="number of epochs", type=int, default=20)
    parser.add_argument("-b", "--batch", help="batch size", type=int, default=64)
    parser.add_argument("-g", "--gpu", help="whether to use GPU for training or not", action="store_true")
    parser.add_argument("-s", "--save", help="where to save trained model", type=str, default="./weights/model.pt")
    args = parser.parse_args()
    
    
    dataset = DisCoTex(root=Path('./data'), batch_size=args.batch, tokenizer=args.tokenizer)
    if args.model == "SR":
        model = SentRegLSTM(
            vocab_size=dataset.tokenizer.get_vocab_size(),
            embed_dim=256,
            hidden_size=256,
            num_layers=4,
            bidirectional=True,
            padd_index=dataset.tokenizer.get_pad_idx(),
            dropout=0.3
        )
    elif args.model == "SC":
        model = SentClasLSTM(
            vocab_size=dataset.tokenizer.get_vocab_size(),
            embed_dim=256,
            hidden_size=256,
            num_layers=4,
            bidirectional=True,
            padd_index=dataset.tokenizer.get_pad_idx(),
            dropout=0.3
        )
    
    trainer = Trainer(max_epochs=args.epoch, lr=1e-4, run_on_gpu=True)
    trainer.fit(model, dataset)
    
    if not os.path.exists(Path(args.save).parent):
        os.mkdir(Path(args.save).parent)
    torch.save(model, Path(args.save))
    
