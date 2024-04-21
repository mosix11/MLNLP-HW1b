import torch
import os
import sys
sys.path.append('./src/')
import argparse
from pathlib import Path

from src.DisCoTex import DisCoTex
from src.trainer import Trainer
from src.models import SentenceClassificationLSTM

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="specify model type between \{\}", type=str, default="")
    parser.add_argument("-t", "--tokenizer", help="specify tokenizer type between \{word, bpe\}", type=str, default="word")
    parser.add_argument("-e", "--epoch", help="number of epochs", type=int, default=20)
    parser.add_argument("-b", "--batch", help="batch size", type=int, default=64)
    parser.add_argument("-g", "--gpu", help="whether to use GPU for training or not", type=bool, action="store_true")
    args = parser.parse_args()
    
    
    dataset = DisCoTex(root=Path('./data'), batch_size=args.batch, tokenizer=args.tokenizer)
    if args.model == "ss":
        pass
    elif args.model == "":
        model = SentenceClassificationLSTM(
            vocab_size=dataset.get_vocab_size(),
            embed_dim=64,
            hidden_size=64,
            num_layers=2,
            bidirectional=True,
            padd_index=dataset.pad_idx,
            dropout=0.3
        )
    
    trainer = Trainer(max_epochs=args.epoch, lr=1e-4, run_on_gpu=True)
    trainer.fit(model, dataset)
    
    torch.save(model, 'model.pt')
    
