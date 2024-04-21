import torch
import os
import sys
sys.path.append('./src/')
import argparse
from pathlib import Path

from src.DisCoTex import DisCoTex
import src.utils as utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--tokenizer", help="specify tokenizer type between \{word, bpe\}", type=str, default="word")
    parser.add_argument("-g", "--gpu", help="whether to use GPU for training or not", type=bool, action="store_true")
    args = parser.parse_args()
    
    dataset = DisCoTex(root=Path('./data'), batch_size=args.batch, tokenizer=args.tokenizer)
    model = torch.load('model.pt')
    if args.gpu:
        model.to(utils.get_gpu_device())
    
    test_loader = dataset.get_test_dataloader()
    preds, labels = [], []
    for i, batch in enumerate(test_loader):
        _, predictions = model.predict(batch[0], batch[1])
        preds.append(*predictions)
        labels.append(*batch[2])
        
    