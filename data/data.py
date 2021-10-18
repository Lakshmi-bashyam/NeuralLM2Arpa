import pytorch_lightning as pl
import torch

from torch.utils.data import random_split, DataLoader
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from sklearn.model_selection import train_test_split


class NgramDataLoader(pl.LightningDataModule):
    def __init__(self, n_gram=3, batch_size=32):
        #Define required parameters here
        self.data = None
        self.n_gram = n_gram
        self.batch_size = batch_size
    
    def prepare_data(self):
        # Define steps that should be done
        # on only one GPU, like getting data.
        with open('final_corpus.txt') as f:
            self.data = f.read().split('\n')
        with open('test.txt') as f_test:
            self.test = f_test.read().split('\n')

        self.tokenizer = get_tokenizer("basic_english")
        self.vocab = build_vocab_from_iterator(map(self.tokenizer, self.data), specials=['<unk>'])
        self.vocab.set_default_index(self.vocab['<unk>'])

    @staticmethod
    def tokenize(vocab, data, tokenizer):
        return([torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in data])

    def setup(self, stage=None):
        # Define steps that should be done on 
        # every GPU, like splitting data, applying
        # transform etc.
        if stage in (None, 'fit'):
            self.train, self.val = train_test_split(self.tokenize(self.vocab, self.data, self.tokenizer), test_size = 0.2)
        if stage == 'test':
            self.test = self.tokenize(self.vocab, self.test, self.tokenizer)

    
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