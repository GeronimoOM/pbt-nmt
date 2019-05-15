from collections import Counter

import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import text_to_word_sequence


def train_test_split(data, split, seed=None):
    n_total = len(data)
    n_train = int(n_total * split)
    np.random.seed(seed)
    index = np.random.permutation(n_total)
    train_index, test_index = index[:n_train], index[n_train:]
    train = [data[i] for i in train_index]
    test = [data[i] for i in test_index]
    return train, test


def read_text(path):
    filters = '0123456789!"„“#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    with open(path, 'r') as file:
        return [[w.replace("'", '') for w in text_to_word_sequence(t, filters=filters)]
                for t in file]


def encode_text(train, test, is_source, maxlen):
    counter = Counter(w for t in train for w in t)
    vocab = set([w for w, c in counter.items() if c >= 5])
    tokenizer = Tokenizer([[w for w in t if w in vocab] for t in train])

    if is_source:
        train = text2sequences(tokenizer, train, padding_type='pre', maxlen=maxlen)
        test = text2sequences(tokenizer, test, padding_type='pre', maxlen=maxlen)
    else:
        train = text2sequences(tokenizer, train, maxlen=maxlen)
        test = text2sequences(tokenizer, test, maxlen=maxlen)
    return train, test, tokenizer


def text2sequences(tokenizer, text, padding_type='post', maxlen=None):
    encoded_text = tokenizer.texts_to_sequences(text)
    sequences = pad_sequences(encoded_text, maxlen=maxlen, padding=padding_type, truncating=padding_type)
    return sequences


def load_nmt(src_path, tar_path, maxlen, split, seed):
    src_text = read_text(src_path)
    tar_text = read_text(tar_path)

    src_train_text, src_test_text = train_test_split(src_text, split, seed)
    tar_train_text, tar_test_text = train_test_split(tar_text, split, seed)

    src_train, src_test, src_tokenizer = encode_text(src_train_text, src_test_text, is_source=True, maxlen=maxlen)
    tar_train, tar_test, tar_tokenizer = encode_text(tar_train_text, tar_test_text, is_source=False, maxlen=maxlen)

    return src_train, src_test, src_tokenizer, tar_train, tar_test, tar_tokenizer


class Tokenizer:
    PAD, PAD_TOK = 0, '<pad>'
    UNK, UNK_TOK = 1, '<unk>'
    BOS, BOS_TOK = 2, '<bos>'

    def __init__(self, texts):
        self.word2idx = {}
        self.idx2word = []

        for word in [Tokenizer.PAD_TOK, Tokenizer.UNK_TOK, Tokenizer.BOS_TOK]:
            self._add_word(word)

        for text in texts:
            for word in text:
                self._add_word(word)

    def _add_word(self, word):
        if word not in self.word2idx:
            idx = len(self)
            self.idx2word.append(word)
            self.word2idx[word] = idx

    def __len__(self):
        return len(self.idx2word)

    def texts_to_sequences(self, texts):
        return [[Tokenizer.BOS] + [self.word2idx.get(word, Tokenizer.UNK) for word in text] for text in texts]

    def sequences_to_texts(self, seqs):
        return [[self.idx2word[idx] for idx in seq] for seq in seqs]


