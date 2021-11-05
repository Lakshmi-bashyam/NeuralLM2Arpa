import pytorch_lightning as pl
import torch
from torch import LongTensor, IntTensor
import os
from torch._C import dtype
from torch.functional import Tensor

from torch.utils.data import random_split, DataLoader, IterableDataset
from torch.utils.data.dataset import Dataset
from sklearn.model_selection import train_test_split
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from torch.nn.utils.rnn import pad_sequence 
from nltk.util import ngrams

class SequenceDataLoader(pl.LightningDataModule):
    def __init__(self, batch_size):
        #Define required parameters here
        super().__init__()
        self.data = None
        self.batch_size = batch_size
    
    def prepare_data(self, root_dir='data'):
        # Define steps that should be done
        # on only one GPU, like getting data.

        with open(os.path.join(root_dir, 'gen.txt')) as f:
            self.data = f.read().split('\n')
        with open(os.path.join(root_dir, 'test_ngram.txt')) as f_test:
            self.test = f_test.read().split('\n')

        self.tokenizer = get_tokenizer("basic_english")
        self.vocab = build_vocab_from_iterator(map(self.tokenizer, self.data))
        self.vocab.append_token('<unk>')
        self.vocab.append_token('<pad>')
        self.vocab.set_default_index(self.vocab['<unk>'])

    def tokenize(self, data):
        # Define the method to tokenize (ngram, full sentence) 
        # and encode text
        result = []
        for item in data:
            item = self.tokenizer(item)
            if len(item) > 1:
                result.append(torch.tensor([self.vocab(item[:-1]), self.vocab(item[1:])]))
        return result
    
    def test_tokenize(self, data):
        result = []
        for item in data:
            item = self.tokenizer(item)
            # if len(item) > 1:
            result.append(torch.tensor(self.vocab(item)))
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
            self.test = self.test_tokenize(self.test)
            # self.test = self.tokenize(self.test)

    
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
        lm_test = DataLoader(self.test, batch_size=self.batch_size, collate_fn=self.pad_collate_test, shuffle=False)

        return lm_test

class NgramDataLoader(pl.LightningDataModule):
    def __init__(self, batch_size=32, ngram=3):
        #Define required parameters here
        super().__init__()
        self.data = None
        self.ngram = ngram
        self.batch_size = batch_size
    
    def prepare_data(self, root_dir='data'):
        # Define steps that should be done
        # on only one GPU, like getting data.

        with open(os.path.join(root_dir, 'final_corpus.txt')) as f:
            self.data = f.read().split('\n')
        with open(os.path.join(root_dir, 'test.txt')) as f_test:
            self.test = f_test.read().split('\n')


        self.tokenizer = get_tokenizer("basic_english")
        self.vocab = build_vocab_from_iterator(map(self.tokenizer, self.data))
        self.vocab.append_token('<unk>')
        self.vocab.set_default_index(self.vocab['<unk>'])

    def tokenize(self, data):
        # Define the method to tokenize (ngram) 
        # and encode text
        result = []
        for item in data:
            item = self.vocab(self.tokenizer(item))
            if len(item) >= self.ngram:
                ngram_list = list(ngrams(item, self.ngram))
                for ngram in ngram_list:
                    result.append([torch.tensor(ngram[:-1], dtype=torch.long), torch.tensor(ngram[-1], dtype=torch.long)])
        return result

    def setup(self, stage=None):
        # Define steps that should be done on 
        # every GPU, like splitting data, applying
        # transform etc.
        if stage in (None, 'fit'):
            data = self.tokenize(self.data)
            total_length = len(data)
            train_length = int(0.8*total_length)
            val_length = int(total_length - train_length)
            self.train, self.val = random_split(data, [train_length, val_length])
        
        if stage == 'test':
            self.test = self.tokenize(self.test)

    
    def train_dataloader(self):
        # Return DataLoader for Training Data here
        lm_train = DataLoader(self.train, batch_size=self.batch_size)
        return lm_train
    
    def val_dataloader(self):
        # Return DataLoader for Validation Data here
        lm_val = DataLoader(self.val, batch_size=self.batch_size)
        return lm_val
    
    def test_dataloader(self):
        # Return DataLoader for Testing Data here
        lm_test = DataLoader(self.test, batch_size=self.batch_size)
        return lm_test



if __name__ == "__main__":
    # obj = NgramDataLoader(ngram=2)
    # obj.prepare_data()
    # obj.setup('fit')
    # for i,j in obj.train_dataloader():
    #     print(i, j)
    #     break

    obj = SequenceDataLoader()
    obj.prepare_data()
    obj.setup('fit')
    for i,j, ilen, jlen in obj.train_dataloader():
        print(i, j)
        break