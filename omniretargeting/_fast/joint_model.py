"""One maximum-excess elastic residual per collision pair, many witness rows."""
import numpy as np


def pair_indices(keys):
    identities = {}
    return np.array([identities.setdefault(key, len(identities)) for key in keys], int)


def maxima(raw, pairs):
    count = int(pairs.max(initial=-1)) + 1
    values = np.zeros(count)
    np.maximum.at(values, pairs, raw)
    return values


def worst_rows(raw, pairs):
    return np.array([int(ids[np.argmax(raw[ids])]) for pair in range(int(pairs.max(initial=-1)) + 1)
                     if len(ids := np.flatnonzero(pairs == pair))], int)


class JointModel:
    def __init__(self, raw, matrix, pairs, sigma):
        self.raw, self.matrix, self.pairs, self.sigma = raw.copy(), matrix.copy(), pairs.copy(), sigma
        self.worst = worst_rows(raw, pairs)

    def correction(self, step):
        move = self.matrix @ step
        old = maxima(self.raw, self.pairs) / self.sigma
        new = maxima(self.raw + move, self.pairs) / self.sigma
        selected = self.worst[self.raw[self.worst] > 0]
        r, j = self.raw[selected] / self.sigma, move[selected] / self.sigma
        gn_decrease = -2 * float(r @ j) - float(j @ j)
        return float(old @ old - new @ new) - gn_decrease
