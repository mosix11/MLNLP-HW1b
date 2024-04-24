import os
import sys
import collections
from pathlib import Path
import nltk
import sentencepiece as spm
from torchtext.vocab import vocab
from typing import Union

class WordLevelTokenizer():
    
    def __init__(self, str_list: list, pad_token:str = '<pad>', unk_token:str = '<unk>',
                 pad_idx:int = 1, unk_idx:int = 0, extra_tokens:list = []) -> None:
        if str_list == None or len(str_list) == 0:
            raise RuntimeError("Invalid list of strings")

        self.str_list = str_list
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx
        self.extra_tokens = extra_tokens
        
        self._build_vocab()
        
    
    def encode(self, text:str):
        return self._token_to_index(self._tokenize(self._preprocess(text)))
    
    def decode(self, idxs:list[int]):
        return list(filter(lambda x: x!='<pad>', self._index_to_token(idxs)))
    
    # def decode_clean(self, idxs:list[int]):
    #     decoded = self._index_to_token(idxs)
    #     cleaned = filter(lambda x: x!='<pad>', decoded)
    #     return ' '.join(cleaned)
    
    def get_vocabulary(self):
        return self.vocabulary
    
    def get_vocab_size(self):
        return len(self.vocabulary)
    
    def get_pad_token(self):
        return self.pad_token
    
    def get_unk_token(self):
        return self.unk_token
    
    def get_pad_idx(self):
        return self.pad_idx
    
    def get_unk_idx(self):
        return self.unk_idx
    
    def _preprocess(self, text:str):
        text = text.strip().replace('\u202f', ' ').replace('\xa0', ' ').replace('\n', ' ').replace('’', '\'')
        out = []
        for i, char in enumerate(text.lower()):
            if char in ',.!?\'' and i > 0 and i < len(text) - 1:
                if text[i-1] != ' ' and text[i+1] != ' ':
                    out.append(' ' + char + ' ')
                    continue
                elif text[i-1] != ' ':
                    out.append(' ' + char)
                    continue
                elif text[i+1] != ' ':
                    out.append(char + ' ')
            out.append(char)
        return ''.join(out)
    
    def _tokenize(self, text:str):
        tokenized_txt = text.split(' ')
        return tokenized_txt
    
    def _build_vocab(self):
        tokenized_set = []
        for sample in self.str_list:
            tokenized_set.append(self._tokenize(self._preprocess(sample)))
        counter = dict(collections.Counter(token for paragraph in tokenized_set for token in paragraph).most_common())
        vocabulary = vocab(counter, min_freq=1, specials=[self.unk_token, self.pad_token, *self.extra_tokens])
        vocabulary.set_default_index(vocabulary([self.unk_token])[0])
        self.vocabulary = vocabulary
        self.pad_idx = self.vocabulary[self.pad_token]
        self.unk_idx = self.vocabulary[self.unk_token]
        
    def _index_to_token(self, data: Union[list[int], list[list[int]]]):
        if isinstance(data, list):
            return self.vocabulary.lookup_tokens(data)
        elif data[0] and isinstance(data[0], list):
            tokens = []
            for sample in data:
                tokens.append(self.vocabulary.lookup_tokens(sample))
            return tokens
        else:
            raise RuntimeError('Invalid input type!!! {}'.format(type(data)))
        
    def _token_to_index(self, data: Union[list[str], list[list[str]]]):
        if isinstance(data, list):
            return self.vocabulary(data)
        elif data[0] and isinstance(data[0], list):
            indexed_data = []
            for sample in data:
                indexed_data.append(self.vocabulary(sample))
            return indexed_data
        else:
            raise RuntimeError('Invalid input type!!! {}'.format(type(data)))


class BPE():
    
    def __init__(self, str_list: list, pad_token:str = '<pad>', unk_token:str = '<unk>',
                 pad_idx:int = 1, unk_idx:int = 0, root_dir:Path = Path('./spm')):
        if not os.path.exists(root_dir):
            os.mkdir(root_dir)
        self.root_dir = root_dir
        self.train_file_dir = self.root_dir / 'train.txt'
        self.saving_model_prefix = self.root_dir / 'it_bpe'
        self.loading_model_dir = self.root_dir / 'it_bpe.model'
        if str_list == None or len(str_list) == 0:
            raise RuntimeError("Invalid list of strings")
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx
        nltk.download('punkt')
        self._save_raw_data(str_list)
        
        
    def _save_raw_data(self, str_list):
        txt = ''
        for paragraph in str_list:
            sentences = nltk.tokenize.sent_tokenize(paragraph)  # Tokenize the paragraph into sentences
            for sentence in sentences:
                if len(sentence.strip()) <= 2:
                    continue
                txt += sentence.strip() + '\n'
        txt = txt[:-1]
        with open(self.train_file_dir, 'w', encoding="utf-8") as f:
            f.write(txt)
    
    def train(self):
        spm.SentencePieceTrainer.train(
            input=self.train_file_dir,       # path to your text file containing the dataset
            # input_format='tsv',
            model_prefix=self.saving_model_prefix,             # prefix for the output model files
            vocab_size=5000,                   # chosen vocabulary size
            model_type='bpe',                  # using Byte Pair Encoding model type
            character_coverage=0.9995,         # nearly full coverage of the character set
            normalization_rule_name='nmt_nfkc_cf',  # Normalization in NFKC_CF (Compatibility Decomposition, followed by Canonical Composition)
            user_defined_symbols=[],           # add any special symbols that might be relevant (e.g., "@", "#", etc.)
            max_sentence_length=2048,          # Maximum length of sentence; adjust based on your specific needs
            shuffle_input_sentence=True,       # shuffle sentences to train in different orders
            remove_extra_whitespaces=True,     # Removes leading, trailing, and duplicate internal whitespace
            unk_piece=self.unk_token,
            pad_piece=self.pad_token,
            unk_id=self.unk_idx,
            pad_id=self.pad_idx,
            bos_id=-1,
            eos_id=-1,
            input_sentence_size=1000000,       # number of sentences to use from the input for training
            num_threads=os.cpu_count()                      # adjust based on your machine's capabilities
        )
        
    def encode(self, text: Union[str, list[str]]):
        return self.sp.Encode(text)
    
    def decode(self, idxs: Union[list[int], list[list[int]]]):
        print(type(idxs), type(idxs[0]))
        return self.sp.Decode(idxs)
    
    def load(self):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(str(self.loading_model_dir))
        
    def get_vocabulary(self):
        vocab = [[self.sp.IdToPiece(idx), idx] for idx in range(self.sp.GetPieceSize())]
        return vocab
    
    def get_vocab_size(self):
        return len([[self.sp.IdToPiece(idx), idx] for idx in range(self.sp.GetPieceSize())])
    
    def get_pad_token(self):
        return self.pad_token
    
    def get_unk_token(self):
        return self.unk_token
    
    def get_pad_idx(self):
        return self.pad_idx
    
    def get_unk_idx(self):
        return self.unk_idx

# class Vocab():
#     """Vocabulary for text."""
#     def __init__(self, tokens=[], min_freq=0, reserved_tokens=[]):
#         # Flatten a 2D list if needed
#         if tokens and isinstance(tokens[0], list):
#             tokens = [token for line in tokens for token in line]
#         counter = collections.Counter(tokens)

#         self.token_freqs = sorted(counter.items(), key=lambda x: x[1],
#                                   reverse=True)
#         # The list of unique tokens
#         self.idx_to_token = list(sorted(set(['<unk>'] + reserved_tokens + [
#             token for token, freq in self.token_freqs if freq >= min_freq])))
#         self.token_to_idx = {token: idx
#                              for idx, token in enumerate(self.idx_to_token)}

#     def __len__(self):
#         return len(self.idx_to_token)
    
#     def __getitem__(self, tokens):
#         if not isinstance(tokens, (list, tuple)):
#             return self.token_to_idx.get(tokens, self.unk)
#         return [self.__getitem__(token) for token in tokens]
    
#     def to_tokens(self, indices):
#         if hasattr(indices, '__len__') and len(indices) > 1:
#             return [self.idx_to_token[int(index)] for index in indices]
#         return self.idx_to_token[indices]

#     @property
#     def unk(self):  # Index for the unknown token
#         return self.token_to_idx['<unk>']
    