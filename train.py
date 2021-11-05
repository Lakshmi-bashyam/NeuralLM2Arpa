from absl import app, flags, logging
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl

from data.data import SequenceDataLoader
from model import LSTMRegressor
import torch
import torch.nn as nn
import sh
import os



flags.DEFINE_boolean('debug', False, '')
flags.DEFINE_integer('epochs', 9, '')
flags.DEFINE_integer('batch_size', 32, '')
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

# sh.rm('-r', '-f', 'logs')
# sh.mkdir('logs')

def main(_):
    data_module = SequenceDataLoader(FLAGS.batch_size)
    data_module.prepare_data('data')
    vocab = data_module.vocab
    vocab_size = len(vocab)
    padding_idx = vocab['<pad>']

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
    # trainer = pl.Trainer(callbacks=[checkpoint_callback])
    trainer.fit(model, data_module)
    print(checkpoint_callback.best_model_path)
    checkpoint = torch.load(checkpoint_callback.best_model_path)
    model.load_state_dict(checkpoint['state_dict'])
    # trainer.save_checkpoint("model.ckpt")
    

    # model = LSTMRegressor.load_from_checkpoint('model.ckpt')
    # print(model.learning_rate)
    trainer.test(model, data_module)




if __name__ == '__main__':
    app.run(main)
