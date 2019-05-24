import numpy as np

from model import define_nmt
from nmt_utils import nmt_train_generator, bleu_score_enc_dec
from data_utils import load_nmt
from pbt import PBT
from members import Member
from train_utils import BleuLogger
from wmt import load_wmt

if __name__ == '__main__':
    np.random.seed(42)

    en_train, en_test, en_tokenizer, de_train, de_test, de_tokenizer = load_wmt(split=0.5)
    en_vocab_size, de_vocab_size = len(en_tokenizer), len(de_tokenizer)

    batch_size = 64
    val_size = 3200
    en_train_t, en_train_v = en_train[val_size:], en_train[:val_size]
    de_train_t, de_train_v = de_train[val_size:], de_train[:val_size]

    # model parameters
    hidden_size = 96
    embedding_size = 100
    timesteps = 30

    # hyperparameters
    lr = 0.001
    dropout = 0.3

    # baseline
    model, encoder_model, decoder_model = define_nmt(
        hidden_size, embedding_size,
        timesteps, en_vocab_size, de_vocab_size, dropout, lr)
    '''
    train_generator = nmt_train_generator(en_train_t, de_train_t, de_vocab_size, batch_size)
    
    eval_every = 100

    loss_logger = LossLogger()
    bleu_logger = BleuLogger((en_train_v, de_train_v), eval_every, batch_size,
                             de_vocab_size, encoder_model, decoder_model)

    model.fit_generator(train_generator, steps_per_epoch=en_train.shape[0]//batch_size,
                        callbacks=[loss_logger, bleu_logger])
    '''
    # default PBT
    lr_values = np.geomspace(1e-4, 1e-1, num=4).tolist()
    dropout_values = np.linspace(0.0, 0.5, num=6).tolist()
    parameters = dict(lr=lr_values, dropout=dropout_values)

    population_size = 16

    def build_member(lr, dropout):
        model, encoder_model, decoder_model = \
            define_nmt(hidden_size, embedding_size, timesteps,
                       en_vocab_size, de_vocab_size, dropout, lr)

        return Member(model, tune_lr=True, use_eval_metric='bleu', custom_metrics={
            'bleu': lambda x, y, _: bleu_score_enc_dec(encoder_model, decoder_model, x, y, batch_size)
        })

    steps_ready = 1000

    generator_fn = lambda x, y, shuffle=True, looping=True: nmt_train_generator(x, y, de_vocab_size, batch_size,
                                                                                shuffle=shuffle, looping=looping)
    pbt = PBT(build_member, population_size, parameters, steps_ready=steps_ready)
    pbt.train(en_train_t, de_train_t, en_train_v, de_train_v, steps=en_train.shape[0]//batch_size,
              eval_every=100, generator_fn=generator_fn)

