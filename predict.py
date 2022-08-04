import json
import os
import pickle
from collections import defaultdict
from operator import itemgetter

import numpy as np
import pytorch_lightning as pl
import torch
from absl import app, flags
from numpy import average
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

from data.data import SequenceDataLoader
from model import LSTMRegressor
from utils import backoff_model, create_matrix

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

def replace_sos_eos(word):
    if word == '<s>':
        word = '<sos>'
    elif word == '</s>':
        word = '<eos>'
    return word

def prob_method():
    data_module = SequenceDataLoader(FLAGS.batch_size)
    data_module.prepare_data('data')
    vocab = data_module.vocab
    lookup = vocab.get_itos()
    vocab_size = len(vocab)
    padding_idx = vocab['<pad>']

    # Intialise model with trained parameters
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
    checkpoint = torch.load('callback_logs/new_data/version1/epoch=8-step=4850.ckpt')
    model.load_state_dict(checkpoint['state_dict'])

    # Set model/data loader to eval mode
    data_module.setup('test')
    model.eval()
    torch.set_grad_enabled(False)
    device = model.device

    # Initialise
    ngram_dict = defaultdict(list)
    lookup = vocab.get_itos()
    reverse_lookup = vocab.get_stoi()

    # backoff LM
    bigram = create_matrix('data/new_lm.arpa', reverse_lookup)
    ngram_save = defaultdict(list)

    # Calculate n-gram probabilities
    for count, batch in enumerate(data_module.test_dataloader()):
        print(count*512)
        # import pdb; pdb.set_trace()
        pred = model.test_step(batch)
        x, y, x_len, y_len = batch
        x = x.view(x.shape[0]*x.shape[1], -1, 1).numpy()
        x = np.repeat(x, repeats=len(lookup), axis=1)
        pred_indices = pred.indices.view(pred.indices.shape[1]*pred.indices.shape[0], pred.indices.shape[2], 1).numpy()
        pred_prob = pred.values.view(pred.values.shape[1]*pred.values.shape[0], pred.values.shape[2], 1).numpy()

        # bigram
        indices = np.dstack((x, pred_indices)).reshape(-1,2)
        prob = pred_prob.reshape(-1,1)

        result = list(zip(indices, prob))       

        with open(f'data/output/ngram_prob_{str(count)}', 'wb') as outfile:
            pickle.dump(result, outfile)
    return ngram_dict

def modified_prob_method():

    data_module = SequenceDataLoader(FLAGS.batch_size)
    data_module.prepare_data('data')
    vocab = data_module.vocab
    lookup = vocab.get_itos()
    reverse_lookup = vocab.get_stoi()
    vocab_size = len(vocab)
    padding_idx = vocab['<pad>']

    # Intialise model with trained parameters
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
    checkpoint = torch.load('callback_logs/new_data/version1/epoch=8-step=4850.ckpt')
    model.load_state_dict(checkpoint['state_dict'])

    # Set model/data loader to eval mode
    data_module.setup('test')
    model.eval()
    torch.set_grad_enabled(False)
    device = model.device


    # initialise
    bigram = defaultdict(list)
    trigram = defaultdict(list)
    
    # backoff LM
    lm = backoff_model(os.path.join('data', 'new_lm.arpa'))
    unigram = lm.ngrams[1][()]
    unigram['<sos'] = unigram['<s>']
    unigram['<eos'] = unigram['</s>']
    unigram.pop('<s>')
    unigram.pop('</s>')
    lm_vocab = list(unigram.keys())
    
    # bigram
    for word_pred, val  in lm.ngrams[2].items():
        word = replace_sos_eos(word_pred[0])
        if word in reverse_lookup:
            pred = model.test_step(
                (torch.tensor([[reverse_lookup[word]]]), 
                None, 
                torch.tensor([1], dtype=torch.int32), 
                None)
                )
            for prob, word_next in zip(pred.values[0], pred.indices[0]):
                word_next = replace_sos_eos(lookup[word_next])
                if word_next in val:
                    bigram[((word,), word_next)].append(prob)
    
    bigram_average = {}
    for k,v in bigram.items():
        bigram_average[k] = average(v)
    with open('data/output/bigram', 'wb') as f:
        pickle.dump(bigram_average, f)

    # trigram
    for word_pred, val  in lm.ngrams[3].items():
        word = (replace_sos_eos(word_pred[0]), replace_sos_eos(word_pred[1]))
        
        if (word[0] in reverse_lookup) and (word[1] in reverse_lookup):
            pred = model.test_step(
                (torch.tensor([[reverse_lookup[i] for i in word]]), 
                None, 
                torch.tensor([2], dtype=torch.int32), 
                None)
                )
            for prob, word_next in zip(pred.values[-1], pred.indices[-1]):
                word_next = replace_sos_eos(lookup[word_next])
                if word_next in val:
                    trigram[((word), word_next)].append(prob)

    trigram_average = {}
    for k,v in trigram.items():
        trigram_average[k] = average(v)
    with open('data/output/trigram', 'wb') as f:
        pickle.dump(trigram_average, f)


def main(_):
    seed_everything(42, workers=True)  
    # prob_method()
    modified_prob_method()

    

if __name__ == '__main__':
    app.run(main)
