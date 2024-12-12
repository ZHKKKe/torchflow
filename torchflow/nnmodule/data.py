import math
import random
import itertools
import numpy as np
import torch


class FiniteBatchSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, indexes, batch_size, shuffle):
        self.indexes = indexes
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.batches = len(self.indexes) // self.batch_size
        if len(self.indexes) % self.batch_size != 0:
            self.batches += 1

    def __iter__(self):
        iterator = self._iterate(self.indexes)
        return (batch for batch in self._group(iterator, self.batch_size))

    def __len__(self):
        return self.batches

    def _group(self, iterable, n):
        args = [iter(iterable)] * n
        return zip(*args)

    def _iterate(self, indices):
        if self.shuffle:
            return np.random.permutation(indices)
        else:
            return indices


class InfiniteBatchSampler(FiniteBatchSampler):
    def __init__(self, indexes, batch_size, shuffle):
        super().__init__(indexes, batch_size, shuffle)

    def _iterate(self, indices):
        def infinite_shuffles():
            while True:
                if self.shuffle:
                    yield np.random.permutation(indices)
                else:
                    yield indices

        return itertools.chain.from_iterable(infinite_shuffles())


class DistributedInfiniteSampler(torch.utils.data.distributed.DistributedSampler):
    def __init__(self, dataset, num_replicas=None,
                 rank=None, shuffle=True, seed=0, drop_last=False):
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)

    def __iter__(self):
        return itertools.chain.from_iterable(self._iterate())

    def _iterate(self):
        while True:
            if self.shuffle:
                g = torch.Generator()
                g.manual_seed(self.seed + random.randint(0, np.iinfo(np.int32).max))
                indices = torch.randperm(len(self.dataset), generator=g).tolist()
            else:
                indices = list(range(len(self.dataset)))

            if not self.drop_last:
                padding_size = self.total_size - len(indices)
                if padding_size <= len(indices):
                    indices += indices[:padding_size]
                else:
                    indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
            else:
                indices = indices[:self.total_size]
            assert len(indices) == self.total_size

            rank_indices = indices[self.rank:self.total_size:self.num_replicas]

            assert len(rank_indices) == self.num_samples
            yield np.random.permutation(rank_indices)
