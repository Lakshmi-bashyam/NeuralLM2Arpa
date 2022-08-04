import os

import pytorch_lightning as pl
import torch
from nltk.util import ngrams
from torch import IntTensor
from torch.functional import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, random_split
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


class SequenceDataLoader(pl.LightningDataModule):
    def __init__(self, batch_size):
        #Define required parameters here
        super().__init__()
        self.data = None
        self.batch_size = batch_size
    
    def prepare_data(self, root_dir='data'):
        # Define steps that should be done
        # on only one GPU, like getting data.

        with open(os.path.join(root_dir, 'final_corpus.txt')) as f:
            self.data = f.read().split('\n')

        self.tokenizer = get_tokenizer("basic_english")
        self.vocab = build_vocab_from_iterator(map(self.tokenizer, self.data))
        self.vocab.append_token('<unk>')
        self.vocab.append_token('<pad>')
        self.vocab.append_token('<sos>')
        self.vocab.append_token('<eos>')
        self.vocab.set_default_index(self.vocab['<unk>'])

    def tokenize(self, data):
        # Define the method to tokenize (ngram, full sentence) 
        # and encode text
        result = []
        for item in data:
            item = self.tokenizer(item)
            item.insert(0, '<sos>')
            item.append('<eos>')
            result.append(torch.tensor([self.vocab(item[:-1]), self.vocab(item[1:])]))
            
        return result
    
    def pad_collate(self, batch):
        # add <pad> tokens to match maximum length in batch
        (xx, yy) = zip(*batch)
        x_lens = IntTensor([len(x) for x in xx])
        y_lens = IntTensor([len(yy) for y in yy])

        xx_pad = pad_sequence(xx, batch_first=True, padding_value=self.vocab['<pad>'])
        yy_pad = pad_sequence(yy, batch_first=True, padding_value=self.vocab['<pad>'])

        x_lens, perm_idx = x_lens.sort(0, descending=True)
        y_lens = x_lens
        xx_pad = xx_pad[perm_idx]
        yy_pad = yy_pad[perm_idx]

        return xx_pad, yy_pad, x_lens, y_lens

    def pad_collate_test(self, batch):
        # add <pad> tokens to match maximum length in batch
        xx = batch
        x_lens = IntTensor([len(x) for x in xx])
        # y_lens = IntTensor([1 for y in yy])

        xx_pad = pad_sequence(xx, batch_first=True, padding_value=self.vocab['<pad>'])
        # yy_pad = pad_sequence(yy, batch_first=True, padding_value=self.vocab['<pad>'])

        x_lens, perm_idx = x_lens.sort(0, descending=True)
        # y_lens = x_lens
        xx_pad = xx_pad[perm_idx]
        # yy_pad = yy[perm_idx]

        return xx_pad, x_lens

    def setup(self, stage=None):
        # Define steps that should be done on 
        # every GPU, like splitting data, applying
        # transform etc.
        if stage in (None, 'fit'):
            total_length = len(self.tokenize(self.data))
            train_length = int(0.8*total_length)
            val_length = int(total_length - train_length)
            self.train, self.val = random_split(self.tokenize(self.data), [train_length, val_length])
        if stage == 'test':
            self.test = self.tokenize(self.data)

    
    def train_dataloader(self):
        # Return DataLoader for Training Data here
        lm_train = DataLoader(self.train, batch_size=self.batch_size, collate_fn=self.pad_collate, shuffle=True, drop_last=True)
        return lm_train
    
    def val_dataloader(self):
        # Return DataLoader for Validation Data here
        lm_val = DataLoader(self.val, batch_size=self.batch_size, collate_fn=self.pad_collate, shuffle=False, drop_last=True)
        return lm_val
    
    def test_dataloader(self):
        # Return DataLoader for Testing Data here
        # lm_test = DataLoader(self.test, batch_size=self.batch_size, collate_fn=self.pad_collate, shuffle=False, drop_last=False)
        lm_test = DataLoader(self.test, batch_size=self.batch_size, collate_fn=self.pad_collate, shuffle=False)

        return lm_test


if __name__ == "__main__":

    obj = SequenceDataLoader(1)
    obj.prepare_data()
    obj.setup('test')
    for i,j, ilen, jlen in obj.test_dataloader():
        print(i, j)
