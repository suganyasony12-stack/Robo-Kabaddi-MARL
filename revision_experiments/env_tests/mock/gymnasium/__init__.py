import numpy as np
from . import spaces
class Env:
    metadata = {}
    def reset(self, seed=None, options=None):
        if seed is not None or not hasattr(self, '_rng'):
            self._rng = np.random.default_rng(seed)
    @property
    def np_random(self): return self._rng
