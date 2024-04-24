import torch
import os
import sys
import argparse
from pathlib import Path

from src import DisCoTex
import src.utils as utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--tokenizer", help="specify tokenizer type between \{word, bpe\}", type=str, default="word")
    parser.add_argument("-g", "--gpu", help="whether to use GPU for training or not", action="store_true")
    parser.add_argument("-l", "--load", help="where to load the weights of the model from", type=str, default="./weights/model.pt")
    parser.add_argument("-b", "--batch", help="test data loader batch size. set 1 to disable batch inference", type=int, default=32)
    args = parser.parse_args()
    
    dataset = DisCoTex(root=Path('./data'), batch_size=args.batch, tokenizer=args.tokenizer)
    model = torch.load(Path(args.load))
    if args.gpu:
        model.to(utils.get_gpu_device())
    
    test_loader = dataset.get_test_dataloader()
    preds, labels = [], []
    for i, batch in enumerate(test_loader):
        if args.gpu:
            batch = [a.to(utils.get_gpu_device()) for a in batch]
        _, predictions = model.predict(batch[0], batch[1])

        preds += predictions.tolist()
        labels += batch[2].tolist()
        
    print(model.accuracy(torch.tensor(preds), torch.tensor(labels)))
    
    