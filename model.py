import pytorch_lightning as pl
import torch
from torch import LongTensor, IntTensor
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Embedding, LSTM
from torch.nn.modules import padding
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
from sklearn.metrics import accuracy_score, f1_score


from data.data import SequenceDataLoader

class LSTMRegressor(pl.LightningModule):
    '''
    Standard PyTorch Lightning module:
    https://pytorch-lightning.readthedocs.io/en/latest/lightning_module.html
    '''
    def __init__(self, 
                 vocab_size,
                 emb_size, 
                 hidden_size, 
                 seq_len,
                 padding_idx,
                 batch_size,
                 num_layers, 
                 dropout, 
                 learning_rate):
        super(LSTMRegressor, self).__init__()
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.padding_idx = padding_idx
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.criterion = nn.CrossEntropyLoss(ignore_index=padding_idx)
        self.learning_rate = learning_rate


        self.embed = Embedding(vocab_size, emb_size, padding_idx=padding_idx)
        self.lstm = nn.LSTM(input_size=emb_size, 
                            hidden_size=hidden_size,
                            num_layers=num_layers, 
                            dropout=dropout, 
                            batch_first=True)
        # self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, vocab_size)
        # self.hidden = None
        
    def forward(self, x, x_len):
        # self.hidden = self.init_hidden(self.batch_size)
        embedded = self.embed(x)
        packed_input = pack_padded_sequence(embedded, x_len.cpu(), batch_first=True)
        packed_output, self.hidden = self.lstm(packed_input, self.hidden)
        output, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
        y_pred = self.linear(output)
        return y_pred

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data.cuda()
        return (Variable(weight.new(self.num_layers, batch_size, self.hidden_size).uniform_()),
                Variable(weight.new(self.num_layers, batch_size, self.hidden_size).uniform_()))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        x, y, x_len, y_len = batch
        self.hidden = self.init_hidden(self.batch_size)
        op = self.forward(x, x_len)
        output_dim = op.shape[-1]
        y_hat = op.view(-1, output_dim)
        y = y.view(-1)
        loss = self.criterion(y_hat, y).mean()
        self.log('train_loss', loss)
        return {'loss': loss, 'log': {'train_loss': loss}}

    def validation_step(self, batch, batch_idx):
        x, y, x_len, y_len = batch
        self.hidden = self.init_hidden(self.batch_size)
        op = self.forward(x, x_len)
        output_dim = op.shape[-1]
        y_hat = op.view(-1, output_dim)
        y = y.view(-1)
        loss = self.criterion(y_hat, y)
        pred = nn.Softmax(dim=1)(y_hat).argmax(dim=1)
        mask = (y != self.padding_idx)
        pred = pred[mask].cpu()
        y = y[mask].cpu()
        acc = accuracy_score(pred, y)
        acc = torch.tensor(acc, dtype=torch.float)
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        return {'loss': loss, 'acc': acc}

    def training_epoch_end(self, outputs):

        loss = torch.stack([o['loss'] for o in outputs], 0).mean()
        # acc = torch.stack([o['acc'] for o in outputs], 0).mean()
        out = {'train_loss': loss, 'train_acc': 0}
        self.log('train_epoch_loss', loss)
        # self.hidden = self.init_hidden(self.batch_size)
        # return {**out, 'log': out}

    def validation_epoch_end(self, outputs):
        loss = torch.stack([o['loss'] for o in outputs], 0).mean()
        acc = torch.stack([o['acc'] for o in outputs], 0).mean()
        out = {'val_loss': loss, 'val_acc': acc}
        return {**out, 'log': out}
    
    def test_step(self, batch, batch_idx):
        x, y, x_len, y_len = batch
        self.hidden = self.init_hidden(self.batch_size)
        op = self.forward(x, x_len)
        output_dim = op.shape[-1]
        y_hat = op.view(-1, output_dim)
        y = y.view(-1)
        loss = self.criterion(y_hat, y)
        pred = nn.Softmax(dim=1)(y_hat).argmax(dim=1)
        mask = (y != self.padding_idx)
        pred = pred[mask].cpu()
        y = y[mask].cpu()
        acc = accuracy_score(pred, y)
        acc = torch.tensor(acc, dtype=torch.float)
        self.log('test_loss', loss)
        self.log('test_acc', acc)
        return {'loss': loss, 'acc': acc}


if __name__ == '__main__':
    data_module = SequenceDataLoader()
    data_module.prepare_data('data')
    vocab = data_module.vocab
    vocab_size = len(vocab)
    padding_idx = vocab['<pad>']

    criterion = nn.CrossEntropyLoss(ignore_index=padding_idx)

    print(vocab_size, padding_idx)
    model = LSTMRegressor(vocab_size, 128, 128, 32, padding_idx, 32, 2, 0.2, 0.02)
    print(model)
    data_module.setup('fit')
    for i, j, ilen, jlen in data_module.train_dataloader():
        op = model.forward(i, ilen)
        output_dim = op.shape[-1]
        op = op.view(-1, output_dim)
        j = j.view(-1)
        loss = criterion(op, j)
        print(loss)
        break