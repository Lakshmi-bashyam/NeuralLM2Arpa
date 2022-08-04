import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Embedding
from torch.nn.functional import softmax
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMRegressor(pl.LightningModule):
    '''
    Standard PyTorch Lightning module:
    https://pytorch-lightning.readthedocs.io/en/latest/lightning_module.html
    '''
    def __init__(self, 
                 vocab_size,
                 emb_size, 
                 lstm_size,
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
        # self.truncated_bptt_steps = 100
        self.lstm_size = lstm_size


        self.embed = Embedding(vocab_size, emb_size, padding_idx=padding_idx)
        self.lstm = nn.LSTM(input_size=emb_size, 
                            hidden_size=lstm_size,
                            num_layers=num_layers, 
                            dropout=dropout, 
                            batch_first=True)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(lstm_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, vocab_size)
        self.relu = nn.ReLU()
        
    def forward(self, x, x_len, hiddens):
        embedded = self.embed(x)
        packed_input = pack_padded_sequence(embedded, x_len.cpu(), batch_first=True)
        packed_output, hiddens = self.lstm(packed_input, hiddens)
        output, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
        rel = self.relu(output)
        dense1 = self.linear(rel)
        drop = self.dropout(dense1)
        y_pred = self.linear2(drop)
        return y_pred

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data.cuda()
        return (Variable(weight.new(self.num_layers, batch_size, self.lstm_size).uniform_()),
                Variable(weight.new(self.num_layers, batch_size, self.lstm_size).uniform_()))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        x, y, x_len, y_len = batch
        hiddens = self.init_hidden(self.batch_size)
        hiddens = (hiddens[0].to(self.device), hiddens[1].to(self.device))
        op = self.forward(x, x_len, hiddens)
        output_dim = op.shape[-1]
        y_hat = op.view(-1, output_dim)
        y = y.view(-1)
        loss = self.criterion(y_hat, y).mean()
        self.log('train_loss', loss)
        return {'loss': loss, 'log': {'train_loss': loss}}

    def validation_step(self, batch, batch_idx):
        x, y, x_len, y_len = batch
        hiddens = self.init_hidden(self.batch_size)
        hiddens = (hiddens[0].to(self.device), hiddens[1].to(self.device))
        op = self.forward(x, x_len, hiddens)
        output_dim = op.shape[-1]
        y_hat = op.view(-1, output_dim)
        y = y.view(-1)
        loss = self.criterion(y_hat, y).mean()
        self.log('val_loss', loss)
        self.log('val_pp', torch.exp(loss))
        return {'loss': loss, 'pp': torch.exp(loss)}

    def training_epoch_end(self, outputs):

        loss = torch.stack([o['loss'] for o in outputs], 0).mean()
        out = {'train_loss': loss, 'train_pp': torch.exp(loss)}
        self.log('train_epoch_loss', loss)

    def validation_epoch_end(self, outputs):
        loss = torch.stack([o['loss'] for o in outputs], 0).mean()
        pp = torch.stack([o['pp'] for o in outputs], 0).mean()
        out = {'val_loss': loss, 'val_pp': pp}
        return {**out, 'log': out}
    
    def test_step(self, batch):
        x, y, x_len, y_len = batch
        hiddens = self.init_hidden(self.batch_size)
        hiddens = (hiddens[0].to(self.device), hiddens[1].to(self.device))
        op = self.forward(x, x_len, hiddens)
        output_dim = op.shape[-1]
        y_hat = op.view(-1, output_dim)
        pred =  torch.topk(softmax(y_hat, dim=1), op.shape[-1])
        return pred

