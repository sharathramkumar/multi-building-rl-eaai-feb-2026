from typing import Iterable
import numpy as np
from abc import ABC, abstractmethod


class BaseGenerator(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def normalize_sample(self) -> float:
        pass

    @abstractmethod
    def get(self) -> float:
        pass


class RepeatingGenerator(BaseGenerator):
    def __init__(self, seq: Iterable, init_index: int = 0) -> None:
        self.seq = seq
        self.ix = init_index
        self.mu = np.array(seq).mean()
        self.std = np.array(seq).std()

    def normalize_sample(self, sample):
        return (sample - self.mu) / self.std

    def get(self, normalized=False):
        sample = self.seq[self.ix]
        self.ix = (self.ix + 1) % len(self.seq)
        if normalized:
            return sample, self.normalize_sample(sample)
        return sample


class ClusterRepeatingGenerator(BaseGenerator):
    def __init__(self, seqs: np.ndarray, init_index: int = 0) -> None:
        self.seqs = seqs
        self.nb, self.tmax = seqs.shape
        self.t_ix = init_index
        self.b_ix = 0
        self.mu = seqs.mean(axis=-1)
        self.std = seqs.std(axis=-1)

    def reset(self):
        self.b_ix = np.random.randint(0, self.nb)

    def normalize_sample(self, sample):
        return (sample - self.mu[self.b_ix]) / self.std[self.b_ix]

    def get(self, normalized=False):
        sample = self.seqs[self.b_ix][self.t_ix]
        self.t_ix = (self.t_ix + 1) % self.tmax
        if normalized:
            return sample, self.normalize_sample(sample)
        return sample
