import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence


class SentenceClassificationLSTM(nn.Module):
    
    def __init__(self, vocab_size:int, embed_dim:int, hidden_size:int, num_layers:int,
                 padd_index:int, dropout:float = 0, bidirectional: bool=False) -> None:
        super().__init__()
        
        
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=padd_index)
        self.rnn = nn.LSTM(input_size=embed_dim, hidden_size=hidden_size, num_layers=num_layers,
                           dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.output_layer = nn.Linear(in_features=hidden_size*2, out_features=4)
        
        
    def forward(self, padded_seqs, lens, H_c=None):
        # padded_seqs indexed and padded batch of sentences of size [B, S]
        # lens sequence valid lengths 
        embeds = self.embed(padded_seqs) # [B, S, H]
        packed_batch = pack_padded_sequence(embeds, lens.cpu(), batch_first=True, enforce_sorted=False)
        
        packed_output, (hidden_states, cell_state) = self.rnn(packed_batch)
        
        hidden = torch.cat((hidden_states[-2,:,:], hidden_states[-1,:,:]), dim = 1)
        logits = self.output_layer(hidden)
        return logits
    
    
    def loss(self, Y_hat, Y, averaged=True):
        Y_hat = Y_hat.reshape((-1, Y_hat.shape[-1]))
        Y = Y.reshape((-1,))
        return F.cross_entropy(
        Y_hat, Y, reduction='mean' if averaged else 'none')
    
    
    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        return l
    
    def validation_step(self, batch):
        Y_hat = self(*batch[:-1])
        return self.loss(Y_hat, batch[-1])
    
    def predict(self, input_seq: torch.LongTensor, valid_len:torch.LongTensor):
        """
        Args:
            input_seq: a tensor of indices
            valid_len: length of the sequence
        Returns:
            A tuple composed of:
            - the logits of each class, 0 and 1
            - the prediction for each sample in the batch
              0 if the sentiment of the sentence is negative, 1 if it is positive.
        """
        self.eval()
        with torch.no_grad():
            # unsqueeze is necessary to add the batch dimension (zero) to the input
            # squeeze remove the added extra dimension
            logits = self(input_seq.unsqueeze(0), valid_len).squeeze()
            predictions = torch.argmax(logits, -1) # computed on the last dimension of the logits tensor
            return logits, predictions
        
        
        
