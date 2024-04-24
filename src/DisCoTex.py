import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

from .tokenizer import WordLevelTokenizer, BPE
from . import utils

from typing import Union
import collections
import random
import jsonlines
import json


import os
import sys
from pathlib import Path



class DisCoTex():
    
    def __init__(self,
                 root:Path = Path('./data'),
                 batch_size:int = 32,
                 num_workers:int = 1,
                 valset_ratio:float = 0.10,
                 tokenizer: str = 'word',
                 device=None) -> None:
        super().__init__()
        
        self.batch_size = batch_size
        self.root = root
        self.num_workers = num_workers
        self.valset_ratio = valset_ratio
        
        self.device = utils.get_cpu_device() if device == None else device
        if tokenizer not in ['word', 'bpe']:
            raise RuntimeError("Invalid tokenizer type!!")

        self.tokenizer_type = tokenizer
        self._load_data()
        self._preprocess_data()
        
        self._init_loaders()
        
    
    def get_train_dataloader(self):
        return self.train_loader
    
    def get_val_dataloader(self):
        return self.val_loader
    
    def get_test_dataloader(self):
        return self.test_loader
    
    def get_tokenizer(self):
        return self.tokenizer

    
    def _init_loaders(self):
        self.train_loader = self._build_dataloader(self.train)
        self.val_loader = self._build_dataloader(self.val)
        self.test_loader = self._build_dataloader(self.test)
    
    
    def _build_dataloader(self, data:list):
        dataset = utils.SimpleListDataset(data)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self._collate_fn, pin_memory=True) 
        
    
    def _preprocess_data(self):
        def _preprocess_wt(jsons:list):
                tokenized_data = [(self.tokenizer.encode(sample[0]), sample[1]) for sample in jsons]
                return tokenized_data
        
        str_list = []
        for sample in self.trainset:
            str_list.append(sample[0])
        if self.tokenizer_type == "word":
            self.tokenizer = WordLevelTokenizer(str_list)
            
        elif self.tokenizer_type == "bpe":
            bpe = BPE(str_list)
            bpe.train()
            bpe.load()
            self.tokenizer = bpe
            
        self.train = _preprocess_wt(self.trainset)
        self.val = _preprocess_wt(self.valset)
        self.test = _preprocess_wt(self.testset)
    
    def _load_data(self):
        data_splits = ['train', 'test']
        file_name = 'discotex-task2-{}-data.jsonl'
        trainset = self._read_jsonl_file(self.root / file_name.format(data_splits[0]))
        self.valset = trainset[:int(len(trainset) * self.valset_ratio)]
        self.trainset = trainset[int(len(trainset) * self.valset_ratio):]    
        self.testset = self._read_jsonl_file(self.root / file_name.format(data_splits[1]))
        self.valset = [(sample['text'], sample['label']) for sample in self.valset]
        self.trainset = [(sample['text'], sample['label']) for sample in self.trainset]
        self.testset = [(sample['text'], sample['label']) for sample in self.testset]
    
    def _collate_fn(self, raw_batch: list[tuple]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # We need these sequence lengths to construct a `torch.nn.utils.rnn.PackedSequence` in the model
        sequence_lengths = torch.tensor([len(sample[0]) for sample in raw_batch], dtype=torch.long)
        
        padded_sequence = pad_sequence(
            [
                torch.tensor(sample[0], dtype=torch.long)
                for sample in raw_batch
            ],
            batch_first=True,
            padding_value=self.tokenizer.get_pad_idx()
        )
        labels = torch.tensor([sample[1] for sample in raw_batch], dtype=torch.long)
        return padded_sequence, sequence_lengths, labels
    
    def _read_jsonl_file(self, path):
        with jsonlines.open(path, 'r') as mfile:
            jsonl_list = []
            for obj in mfile:
                jsonl_list.append(obj)
        return jsonl_list
        

    