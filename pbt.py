import os

import numpy as np

from model import define_nmt
from data_utils import load_nmt
from nmt_utils import nmt_generator, nmt_predict, eval_bleu_score


def load_wmt(data_folder='data', maxlen=30, split=0.5):
    path = os.path.join(data_folder, 'europarl.npz')
    if os.path.exists(path):
        with np.load(path, allow_pickle=True) as data:
            en_train, en_test, en_tokenizer = data['en_train'], data['en_test'], data['en_tokenizer']
            de_train, de_test, de_tokenizer = data['de_train'], data['de_test'], data['de_tokenizer']
            return en_train, en_test, en_tokenizer.item(), de_train, de_test, de_tokenizer.item()

    else:
        en_file = 'europarl-v7.de-en.en'
        de_file = 'europarl-v7.de-en.de'

        en_train, en_test, en_tokenizer, de_train, de_test, de_tokenizer = load_nmt(
            src_path=os.path.join(data_folder, en_file),
            tar_path=os.path.join(data_folder, de_file),
            maxlen=maxlen, split=split, seed=0)

        data = [en_train, en_test, en_tokenizer, de_train, de_test, de_tokenizer]
        np.savez(path, en_train=en_train, en_test=en_test, en_tokenizer=en_tokenizer,
                 de_train=de_train, de_test=de_test, de_tokenizer=de_tokenizer)

    return data


if __name__ == '__main__':
    hidden_size = 96
    embedding_size = 100
    timesteps = 30
    batch_size = 64

    en_train, en_test, en_tokenizer, de_train, de_test, de_tokenizer = load_wmt(split=0.3)
    en_vocab_size, de_vocab_size = len(en_tokenizer), len(de_tokenizer)

    model, encoder_model, decoder_model = define_nmt(hidden_size, embedding_size, timesteps,
                                                     en_vocab_size, de_vocab_size)

    generator = nmt_generator(en_train, de_train, de_vocab_size, batch_size, shuffle=True)
    model.fit_generator(generator, steps_per_epoch=en_train.shape[0]//batch_size, epochs=2)

    print(eval_bleu_score(encoder_model, decoder_model, en_test[:100], de_test[:100], batch_size))


