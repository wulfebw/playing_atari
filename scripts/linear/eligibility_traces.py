
import collections
import numpy as np

class EligibilityTraces(object):
    """
    :description: Dictionary wrapper that clips input values and can update all values
        by a constant multiple (self.decay), discarding those that fall below self.threshold.
    """
    def __init__(self, threshold, decay):
        self.threshold = threshold
        self.decay = decay
        self.ets = collections.Counter()

    def __getitem__(self, k):
        return self.ets[k]

    def __setitem__(self, k, v):
        self.ets[k] = np.clip(v, 0, 1)

    def __contains__(self, k):
        return k in self.ets

    def iteritems(self):
        return self.ets.iteritems()

    def update_all(self):
        for f in self.ets.keys():
            if self.ets[f] < self.threshold:
                del self.ets[f]
            else:
                self.ets[f] = self.ets[f] * self.decay
