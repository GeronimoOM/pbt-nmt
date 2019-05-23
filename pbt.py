
from collections import defaultdict
from typing import List

import numpy as np

from members import Member
from train_utils import batch_generator


class PBT:

    def __init__(self, population: List[Member], steps_ready):
        self.population = population
        self.steps_ready = steps_ready

    def train(self, x_train, y_train, x_val, y_val, steps, eval_every,
              generator_fn=batch_generator):
        train_gen = generator_fn(x_train, y_train)

        results = defaultdict(lambda: [])

        '''
        #metric_names = [stat for metric in ['loss'] + self.population[0].metrics
        #                for stat in _metric_stats(metric)]
        #progbar = Progbar(steps, stateful_metrics=metric_names)
        '''
        for step in range(1, steps + 1):
            x, y = next(train_gen)

            for member in self.population:
                member.step_on_batch(x, y)
                if step % eval_every == 0 or step == steps:
                    member.eval(x_val, y_val, generator_fn)

                if self.ready(member):
                    exploited = self.exploit(member)
                    if exploited:
                        self.explore(member)
                        member.eval(x_val, y_val, generator_fn)

                if step % eval_every == 0 or step == steps:
                    results_mem = {
                        'step': member.steps,
                        'loss': member.loss
                    }
                    for metric, value in member.metrics.items():
                        results_mem[metric] = value
                    for h, v in member.get_hyperparameter_config().items():
                        results_mem[h] = v

                    results[str(member)].append(results_mem)

            '''
            all_losses = results['loss']
            recent_losses = all_losses[-population_size:]
            if recent_losses:
                metric_names = _statistics(recent_losses, 'loss')
                for metric, _ in population[0].metrics.items:
                    metric_names.extend(
                        _statistics(results[metric][-population_size:], metric))
                progbar.update(step, metric_names)

            '''
        return results

    def ready(self, member):
        return member.steps % self.steps_ready == 0

    def exploit(self, member):
        evals = np.array([m.eval_metric() for m in self.population])
        # Lower is better. Top 20% means percentile 20 in losses
        threshold_best, threshold_worst = np.percentile(evals, (20, 80))
        if member.eval_metric() > threshold_worst:
            top_performers = [m for m in self.population
                              if m.eval_metric() < threshold_best]
            if top_performers:
                member.replace_with(np.random.choice(top_performers))
            return True
        else:
            return False

    def explore(self, member):
        for h in member.hyperparameters:
            h.perturb([0.8, 1.2])


def _metric_stats(metric):
    return [m.format(metric) for m in ['min_{}', 'max_{}', 'mean_{}']]


def _statistics(values, suffix):
    min_value = ('min_{}'.format(suffix), min(values))
    max_value = ('max_{}'.format(suffix), max(values))
    mean_value = ('mean_{}'.format(suffix), sum(values) / len(values))
    return [min_value, max_value, mean_value]