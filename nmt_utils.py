import numpy as np
from keras.utils import to_categorical
import nltk

from data_utils import Tokenizer


def nmt_generator(src, tar, tar_vocab_size, batch_size, shuffle=False):
    indices = np.arange(len(src))

    while True:
        if shuffle:
            np.random.shuffle(indices)

        for idx in range(0, indices.shape[0] - batch_size + 1, batch_size):
            batch_index = indices[idx:idx+batch_size]

            input_src, input_tar = src[batch_index], tar[batch_index, :-1]
            output_tar = to_categorical(tar[batch_index, 1:], num_classes=tar_vocab_size)

            yield [input_src, input_tar], output_tar


def nmt_predict(encoder, decoder, src, batch_size):
    indices = np.arange(len(src))
    preds = []

    for idx in range(0, indices.shape[0] - batch_size + 1, batch_size):
        batch_index = indices[idx:idx + batch_size]
        batch_enc_input = src[batch_index]
        batch_dec_input = np.full(batch_size, Tokenizer.BOS)
        batch_preds = []

        encoder_out, encoder_state = encoder.predict(batch_enc_input )
        decoder_state = encoder_state

        cur_index = np.arange(batch_size)
        for t in range(src.shape[1]):
            decoder_pred, decoder_state = \
                decoder.predict([encoder_out[cur_index], decoder_state, np.expand_dims(batch_dec_input, axis=1)])
            decoder_max_pred = np.argmax(decoder_pred, axis=-1)[:, 0]

            next_index = decoder_max_pred != Tokenizer.PAD
            if not any(next_index):
                break

            cur_index = cur_index[next_index]
            decoder_state = decoder_state[next_index]
            batch_dec_input = decoder_max_pred[next_index]

            batch_pred = np.repeat(Tokenizer.PAD, batch_size)
            batch_pred[cur_index] = batch_dec_input
            batch_preds.append(batch_pred)

        batch_preds = np.array(batch_preds).T
        if batch_preds.shape[1] < src.shape[1]:
            pads = np.full((src.shape[1] - batch_preds.shape[1], batch_size), Tokenizer.PAD)
            batch_preds = np.hstack([batch_preds, pads])

        preds.append(batch_preds)

    preds = np.vstack(preds)
    return preds


def bleu_score(target, pred):
    return nltk.translate.bleu_score.sentence_bleu([target], pred)


def eval_bleu_score(encoder, decoder, src, tar, batch_size):
    preds = nmt_predict(encoder, decoder, src, batch_size)
    return np.mean([bleu_score(t, p) for t, p in zip(tar, preds)])

