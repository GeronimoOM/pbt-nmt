import numpy as np
from keras.utils import to_categorical
import nltk

from data_utils import Tokenizer


def nmt_train_generator(src, tar, tar_vocab_size, batch_size=64, shuffle=True):
    indices = np.arange(len(src))

    while True:
        if shuffle:
            np.random.shuffle(indices)

        for idx in range(0, indices.shape[0] - batch_size + 1, batch_size):
            batch_index = indices[idx:idx+batch_size]

            input_src, input_tar = src[batch_index], tar[batch_index, :-1]
            output_tar = to_categorical(tar[batch_index, 1:], num_classes=tar_vocab_size)

            yield [input_src, input_tar], output_tar


def nmt_infer_generator(src, tar, batch_size=64):
    indices = np.arange(len(src))

    for idx in range(0, indices.shape[0] - batch_size + 1, batch_size):
        batch_index = indices[idx:idx + batch_size]

        input_src = src[batch_index]
        input_tar = tar[batch_index, :-1]

        yield input_src, input_tar


def nmt_infer(encoder, decoder, inputs):
    preds = np.full((inputs.shape[1], inputs.shape[0]), Tokenizer.PAD)
    decoder_inputs = np.full(inputs.shape[0], Tokenizer.BOS)

    encoder_out, encoder_state = encoder.predict(inputs)
    decoder_state = encoder_state

    index = np.arange(len(inputs))
    for t in range(inputs.shape[1]):
        decoder_pred, decoder_state = \
            decoder.predict([encoder_out[index], decoder_state, np.expand_dims(decoder_inputs, axis=1)])
        decoder_max_pred = np.argmax(decoder_pred, axis=-1)[:, 0]

        next_index = decoder_max_pred != Tokenizer.PAD
        if not any(next_index):
            break

        index = index[next_index]
        decoder_state = decoder_state[next_index]
        decoder_inputs = decoder_max_pred[next_index]
        preds[t, index] = decoder_inputs

    return preds.T


def bleu_score(y_true, y_pred):
    return nltk.translate.bleu_score.sentence_bleu([y_true], y_pred)


def bleu_score_enc_dec(encoder, decoder, src, tar, batch_size=64):
    n_batches = src.shape[0] // batch_size
    scores = np.zeros(batch_size * n_batches)
    for b, (src, tar) in enumerate(nmt_infer_generator(src, tar, batch_size)):
        preds = nmt_infer(encoder, decoder, src)
        scores[b*batch_size:(b+1)*batch_size] = [bleu_score(t, p) for t, p in zip(tar, preds)]
    return np.mean(scores)