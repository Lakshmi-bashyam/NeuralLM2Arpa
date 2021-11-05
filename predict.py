# from train import FLAGS
from data.data import SequenceDataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from torch.nn.functional import softmax
from model import LSTMRegressor

import torch
from absl import app, flags
from collections import defaultdict
import json

flags.DEFINE_boolean('debug', False, '')
flags.DEFINE_integer('epochs', 9, '')
flags.DEFINE_integer('batch_size', 1, '')
flags.DEFINE_float('lr', 1e-4 , '')
flags.DEFINE_float('momentum', .9, '')
flags.DEFINE_float('dropout', .3, '')
# flags.DEFINE_string('dataset', 'SequenceDataLoader', '')
# flags.DEFINE_string('model', 'bert-base-uncased', '')
flags.DEFINE_integer('seq_length', 32, '')
flags.DEFINE_integer('embedding_size', 512, '')
flags.DEFINE_integer('lstm_size', 128, '')
flags.DEFINE_integer('hidden_size', 128, '')
flags.DEFINE_integer('layers', 2, '')

FLAGS = flags.FLAGS


    # ------------------TO-DO--------------------------
    # 1. Run and load the big model for extracted dataset
    # 2. Put it in a seperate file to run
    # 3. get n-grams from arpa file or nltk
    # 4. write back to arpa
    # 5. Check if anything else has to be done
    # --------------------------------------------------


def main(_):
    data_module = SequenceDataLoader(FLAGS.batch_size)
    data_module.prepare_data('data')
    vocab = data_module.vocab
    lookup = vocab.get_itos()
    vocab_size = len(vocab)
    padding_idx = vocab['<pad>']

    checkpoint_callback = ModelCheckpoint(dirpath="callback_logs/new_data/version1/", monitor='val_loss')
    trainer = pl.Trainer(
        default_root_dir='logs',
        gpus=(1 if torch.cuda.is_available() else 0),
        max_epochs=FLAGS.epochs,
        fast_dev_run=FLAGS.debug,
        logger=pl.loggers.TensorBoardLogger('logs/', name='ATCO_sequence', version=3),
        log_every_n_steps=50,
        callbacks=[checkpoint_callback]
    )
    model = LSTMRegressor(vocab_size, 
                FLAGS.embedding_size, 
                FLAGS.lstm_size,
                FLAGS.hidden_size,
                FLAGS.seq_length,
                padding_idx,
                FLAGS.batch_size,
                FLAGS.layers,
                FLAGS.dropout,
                FLAGS.lr)
    print(model)
    checkpoint = torch.load('callback_logs/new_data/version1/epoch=8-step=11051.ckpt')
    model.load_state_dict(checkpoint['state_dict'])

    # ------------- TO-DO --------------
    # turn on after changing data module
    # -----------------------------------
    # trainer.test(model, data_module) 


    # Calculate ngram probabilities
    data_module.setup('test')
    device = model.device
    ngram_dict = defaultdict(list)
    lookup = vocab.get_itos()

    #   #   Averaging method - not recommended
    # for x, y, x_len, y_len in data_module.test_dataloader():
    #     hiddens =model.init_hidden(1)
    #     hiddens = (hiddens[0].to(device), hiddens[1].to(device))
    #     op = model.forward(x, x_len, hiddens)
    #     output_dim = op.shape[-1]
    #     y_hat = op.view(-1, output_dim)
    #     x = x[0]
    #     # For every word in training corpus
    #     for i, val in enumerate(x):
    #         if i > 0:
    #             h = [lookup[i] for i in x[i-1:i+1]]
    #         else:
    #             h = [lookup[val]]
    #         pred = y_hat[i]
    #         pred = torch.topk(nn.functional.softmax(pred), vocab_size)
    #         for prob, val in zip(*pred):
    #             w = lookup[val]
    #             bi_gram = h[-1] + ' ' + w
    #             prob = float(prob.detach())
    #             ngram_dict[bi_gram].append(prob)
    #             if i > 0:
    #                 tri_gram = h[-2] + ' ' + h[-1] + ' ' + w
    #                 ngram_dict[tri_gram].append(prob)
    #             # print(ngram_dict)
    #         print(ngram_dict)


    # From n-grams
    ngram_to_save = {}
    for x, x_len in data_module.test_dataloader():
        hiddens =model.init_hidden(1)
        hiddens = (hiddens[0].to(device), hiddens[1].to(device))
        op = model.forward(x, x_len, hiddens)
        output_dim = op.shape[-1]
        y_hat = op.view(-1, output_dim)
        pred = y_hat[-1]
        pred = torch.topk(softmax(pred), vocab_size)
        x = x[0]
        h = [lookup[i] for i in x]
        for prob, val in zip(*pred):
            w = lookup[val]
            prob = float(prob.detach())
            if len(h) > 1:
                tri_gram = h[-2] + ' ' + h[-1] + w
                ngram_to_save[tri_gram] = prob
            else:
                bi_gram = h[-1] + ' ' + w
                ngram_to_save[bi_gram] = prob

        break

    with open('data/ngram_prob.json', 'wb') as outfile:
        json.dump(ngram_to_save, outfile)

    

if __name__ == '__main__':
    app.run(main)
