import os
from collections import defaultdict

import numpy as np
import pandas as pd
from keras.utils import Progbar
from sklearn.model_selection import ParameterGrid

from model import define_nmt
from nmt_utils import nmt_train_generator, bleu_score_enc_dec
from data_utils import load_nmt, train_test_split
from pbt.members import Member


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


def _statistics(values, suffix):
    min_value = ('min_{}'.format(suffix), min(values))
    max_value = ('max_{}'.format(suffix), max(values))
    mean_value = ('mean_{}'.format(suffix), sum(values) / len(values))
    return [min_value, max_value, mean_value]


if __name__ == '__main__':
    seed = 42

    # load data
    en_train, en_test, en_tokenizer, de_train, de_test, de_tokenizer = load_wmt(split=0.3)
    en_vocab_size, de_vocab_size = len(en_tokenizer), len(de_tokenizer)

    # model parameters
    hidden_size = 96
    embedding_size = 100
    timesteps = 30

    batch_size = 64
    train_generator = nmt_train_generator(en_train, de_train, de_vocab_size, batch_size)
    val_generator = nmt_train_generator(en_test, de_test, de_vocab_size, batch_size)
    x_val, y_val = next(val_generator)

    # hyperparameter search space
    lr_values = np.geomspace(1e-3, 1e-2, num=4).tolist()
    dropout_values = np.geomspace(1e-10, 1e-1, num=3).tolist()
    param_grid = ParameterGrid(dict(lr=lr_values, dropout=dropout_values))

    # generate population
    pop_size = 8
    population = []
    model_dict = {}

    total_steps = en_train.shape[0] // batch_size
    steps_ready = 1000
    steps_save = 100

    def build_fn(dropout, lr):

        def _build_fn():
            model, encoder_model, decoder_model = define_nmt(hidden_size, embedding_size, timesteps,
                                                             en_vocab_size, de_vocab_size, dropout, lr)
            model_dict[model] = (encoder_model, decoder_model)
            return model

        return _build_fn

    for i in np.linspace(0, len(param_grid) - 1, pop_size):
        h_idx = int(round(i))
        h = param_grid[h_idx]
        member = Member(build_fn(**h), tune_lr=True, steps_ready=steps_ready)
        population.append(member)

    results = defaultdict(lambda: [])
    stateful_metrics = ['min_loss', 'max_loss', 'mean_loss']
    for metric, _ in population[0].eval_metrics:
        stateful_metrics.extend([m.format(metric) for m in ['min_{}', 'max_{}', 'mean_{}']])
    progbar = Progbar(total_steps, stateful_metrics=stateful_metrics)

    for step in range(1, total_steps + 1):
        print('step')
        x, y = next(train_generator)
        for idx, member in enumerate(population):
            member.step_on_batch(x, y)
            loss = member.eval_on_batch(x_val, y_val)

            if member.ready():
                exploited = member.exploit(population)
                if exploited:
                    member.explore()
                    loss = member.eval_on_batch(x_val, y_val)

            if step % steps_save == 0 or step == total_steps:
                results['model_id'].append(str(member))
                results['step'].append(step)
                results['loss'].append(loss)
                results['loss_smoothed'].append(member.loss_smoothed())
                for metric, value in member.eval_metrics:
                    results[metric].append(value)
                for h, v in member.get_hyperparameter_config().items():
                    results[h].append(v)

        # Get recently added losses to show in the progress bar
        all_losses = results['loss']
        recent_losses = all_losses[-pop_size:]
        if recent_losses:
            metrics = _statistics(recent_losses, 'loss')
            for metric, _ in population[0].eval_metrics:
                metrics.extend(
                    _statistics(results[metric][-pop_size:], metric))
            progbar.update(step, metrics)

    results = pd.DataFrame(results)
    print(results)
