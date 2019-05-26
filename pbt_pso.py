from collections import Callable

import numpy as np

from pbt import PbtOptimizer


class PbtPsoOptimizer(PbtOptimizer):

    def __init__(self, build_member: Callable,
                 population_size: int,
                 parameters: dict,
                 steps_ready: int,
                 omega: float = 0.3,
                 phi1: float = 0.6,
                 phi2: float = 0.8):
        super().__init__(build_member, population_size, parameters, steps_ready)
        self.omega = omega
        self.phi1 = phi1
        self.phi2 = phi2

        self.population_best = {m: m.get_hyperparameter_config() for m in self.population}
        self.population_best_score = {m: 0 for m in self.population}
        self.global_best = 0
        self.global_best_score = 0
        self.bounds = {n: (min(values), max(values)) for n, values in parameters.items()}
        bound_diffs = {n: abs(hi - lo) for n, (lo, hi) in self.bounds.items()}
        self.velocity = {m: {n: np.random.uniform(-bound_diffs[n], bound_diffs[n])
                             for n in m.hyperparameter_config.keys()}
                         for m in self.population}

    def exploit(self, member):
        velocity = self.velocity[member]
        member_best = self.population_best[member]
        for n, h in member.hyperparameter_config.items():
            r1, r2 = np.random.uniform(0.0, 1.0, size=2)
            new_vel = self.omega * velocity[n] \
                      + self.phi1 * r1 * (member_best[n] - h.get()) \
                      + self.phi2 * r2 * (self.global_best[n] - h.get())
            velocity[n] = new_vel
            h.set(max(min(h.get() + new_vel, self.bounds[n][1]), self.bounds[n][0]))

        return True

    def explore(self, member):
        pass

    def on_eval(self, member):
        if member.eval_metric() > self.population_best_score[member]:
            self.population_best[member] = member.get_hyperparameter_config()
            self.population_best_score[member] = member.eval_metric()
        if member.eval_metric() > self.global_best_score:
            self.global_best = member.get_hyperparameter_config()
            self.global_best_score = member.eval_metric()


