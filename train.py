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
flags.DEFINE_integer('epochs', 20, '')
flags.DEFINE_integer('batch_size', 32, '')
flags.DEFINE_float('lr', 1e-2 , '')
flags.DEFINE_float('momentum', .9, '')
flags.DEFINE_float('dropout', .2, '')
# flags.DEFINE_string('dataset', 'SequenceDataLoader', '')
# flags.DEFINE_string('model', 'bert-base-uncased', '')
flags.DEFINE_integer('seq_length', 32, '')
flags.DEFINE_integer('embedding_size', 256, '')
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
                FLAGS.hidden_size,
                FLAGS.seq_length,
                padding_idx,
                FLAGS.batch_size,
                FLAGS.layers,
                FLAGS.dropout,
                FLAGS.lr)
    
    if os.path.exists('logs/ATCO_sequenceersion_1/checkpoints/epoch=12-step=6993.ckpt'):
        CKPT_PATH = "logs/"
        checkpoint = torch.load(CKPT_PATH)
        print(checkpoint["hyper_parameters"])
        trainer = pl.Trainer(callbacks=[checkpoint])
        trainer.fit(model)
        checkpoint.best_model_path

    else:
        trainer = pl.Trainer(
            default_root_dir='logs',
            gpus=(1 if torch.cuda.is_available() else 0),
            max_epochs=FLAGS.epochs,
            fast_dev_run=FLAGS.debug,
            logger=pl.loggers.TensorBoardLogger('logs/', name='ATCO_sequence', version=1),
            log_every_n_steps=50
        )
        trainer.fit(model, data_module)
        trainer.save_checkpoint("model.ckpt")
    trainer.test(model, data_module)


if __name__ == '__main__':
    app.run(main)
