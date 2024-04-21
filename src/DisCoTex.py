import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import vocab

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

from tokenizer import WordLevelTokenizer
import utils

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
                 root:Path = Path('../data'),
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
        if tokenizer == 'word':
            self.tokenizer = WordLevelTokenizer().tokenizer
        elif tokenizer == 'bpe':
            self.tokenizer = None
        else:
            raise RuntimeError('Invalid Tokenizer Type')
        self._load_data()
        self._build_vocab(self.train['tokenized'])
        
        self._init_loaders()
        
    
    def get_train_dataloader(self):
        return self.train_loader
    
    def get_val_dataloader(self):
        return self.val_loader
    
    def get_test_dataloader(self):
        return self.test_loader
    
    def get_vocabulary(self):
        return self.vocabulary
    
    def get_vocab_size(self):
        return len(self.vocabulary)
    
    def _init_loaders(self):
        self.train_loader = self._build_dataloader(self._token_to_index(self.train['tokenized']))
        self.val_loader = self._build_dataloader(self._token_to_index(self.val['tokenized']))
        self.test_loader = self._build_dataloader(self._token_to_index(self.test['tokenized']))
    
    
    def _build_dataloader(self, data:list):
        dataset = utils.SimpleListDataset(data)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self._collate_fn, pin_memory=True) 
    
    def _token_to_index(self, data: Union[list, tuple]):
        if data and (isinstance(data[0], list) or (isinstance(data[0], tuple))):
            indexed_data = []
            for sample in data:
                indexed_data.append((self.vocabulary(sample[0]), sample[1]))
            return indexed_data
        else:
            return (self.vocabulary(data[0]), data[1])
            
    def index_to_token(self, data: Union[list, tuple]):
        return self.vocabulary.get_itos(data)
    
    def _build_vocab(self, tokenized_set:list, pad_token:str = '<pad>', unk_token:str = '<unk>', extra_tokens:list[str] = []):
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.extra_tokens = extra_tokens
        counter = dict(collections.Counter(token for paragraph in tokenized_set for token in paragraph[0]).most_common())
        vocabulary = vocab(counter, min_freq=1, specials=[pad_token, unk_token, *extra_tokens])
        vocabulary.set_default_index(vocabulary([unk_token])[0])
        self.vocabulary = vocabulary
        self.pad_idx = self.vocabulary[self.pad_token]
        self.unk_idx = self.vocabulary[self.pad_token]
    
    def _load_data(self):
        data_splits = ['train', 'test']
        file_name = 'discotex-task2-{}-data.jsonl'
        trainset = self._read_jsonl_file(self.root / file_name.format(data_splits[0]))
        valset = trainset[:int(len(trainset) * self.valset_ratio)]
        trainset = trainset[int(len(trainset) * self.valset_ratio):]    
        testset = self._read_jsonl_file(self.root / file_name.format(data_splits[1]))
        
        train_orig, train_tokenized = self._preprocess_data(trainset)
        val_orig, val_tokenized = self._preprocess_data(valset)
        test_orig, test_tokenized = self._preprocess_data(testset)
        
        self.train = {'original':train_orig, 'tokenized':train_tokenized}
        self.val = {'original': val_orig, 'tokenized': val_tokenized}
        self.test = {'original': test_orig, 'tokenized': test_tokenized}
        

    
    def _preprocess_data(self, jsons:list):
        orig_data = [(sample['text'], sample['label']) for sample in jsons]
        tokenized_data = [(self.tokenizer(sample['text']), sample['label']) for sample in jsons]
        return orig_data, tokenized_data    
    
    def _collate_fn(self, raw_batch: list[tuple]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # We need these sequence lengths to construct a `torch.nn.utils.rnn.PackedSequence` in the model
        sequence_lengths = torch.tensor([len(sample[0]) for sample in raw_batch], dtype=torch.long)
        
        padded_sequence = pad_sequence(
            [
                torch.tensor(sample[0], dtype=torch.long)
                for sample in raw_batch
            ],
            batch_first=True,
            padding_value=self.pad_idx
        )
        labels = torch.tensor([sample[1] for sample in raw_batch], dtype=torch.long)
        return padded_sequence, sequence_lengths, labels
    
    def _read_jsonl_file(self, path):
        with jsonlines.open(path, 'r') as mfile:
            jsonl_list = []
            for obj in mfile:
                jsonl_list.append(obj)
        return jsonl_list
        
# ds = DisCoTex()
# dl = ds.get_train_dataloader()
# for batch in dl:
#     paded, lens, labels = batch

    # print(type(lens), lens.shape)
    # print(type(paded), paded.shape)
    # print(type(labels), labels.shape)
    
    