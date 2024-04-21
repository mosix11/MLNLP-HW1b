import os
import sys
import collections


class WordLevelTokenizer():
    
    def __init__(self) -> None:
        pass
        
    def tokenizer(self, text:str):
        return self._tokenize(self._preprocess(text))
    
    def _preprocess(self, text:str):
        # Replace non-breaking space with space
        text = text.strip().replace('\u202f', ' ').replace('\xa0', ' ').replace('\n', ' ').replace('’', '\'')
        # Insert space between words and punctuation marks
        # no_space_punc = lambda char, prev_char: char in ',.!?’\'' and prev_char != ' '
        # out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
        #     for i, char in enumerate(text.lower())]
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

class Vocab():
    """Vocabulary for text."""
    def __init__(self, tokens=[], min_freq=0, reserved_tokens=[]):
        # Flatten a 2D list if needed
        if tokens and isinstance(tokens[0], list):
            tokens = [token for line in tokens for token in line]
        counter = collections.Counter(tokens)

        self.token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                  reverse=True)
        # The list of unique tokens
        self.idx_to_token = list(sorted(set(['<unk>'] + reserved_tokens + [
            token for token, freq in self.token_freqs if freq >= min_freq])))
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}

    def __len__(self):
        return len(self.idx_to_token)
    
    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]
    
    def to_tokens(self, indices):
        if hasattr(indices, '__len__') and len(indices) > 1:
            return [self.idx_to_token[int(index)] for index in indices]
        return self.idx_to_token[indices]

    @property
    def unk(self):  # Index for the unknown token
        return self.token_to_idx['<unk>']
    