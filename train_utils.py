import keras
import numpy as np

from nmt_utils import nmt_train_generator, bleu_score_enc_dec


class LossLogger(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


class BleuLogger(keras.callbacks.Callback):

    def __init__(self, data, eval_every, batch_size, tar_vocab_size, encoder, decoder):
        self.src, self.tar = data
        self.generator = nmt_train_generator(self.src, self.tar,
                                             tar_vocab_size, batch_size)
        self.eval_every = eval_every
        self.batch_size = batch_size
        self.encoder = encoder
        self.decoder = decoder

    def on_train_begin(self, logs={}):
        self.batch = 0
        self.losses = []
        self.scores = []

    def on_batch_end(self, batch, logs={}):
        if (batch + 1) % self.eval_every == 0:
            loss = np.mean([self.model.test_on_batch(*next(self.generator))
                            for _ in range(self.src.shape[0] // self.batch_size)])
            bleu = bleu_score_enc_dec(self.encoder, self.decoder, self.src, self.tar, self.batch_size)
            self.losses.append(loss)
            self.scores.append(bleu)