from random import seed as _set_seed
from numpy.random import default_rng as _default_rng


def set_seed(seed):
    global _global_seed, _np_rng
    _np_rng = _default_rng(seed)
    _set_seed(seed)
    _global_seed = seed

set_seed(None)