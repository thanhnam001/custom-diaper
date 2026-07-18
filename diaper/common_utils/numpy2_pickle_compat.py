#!/usr/bin/env python3

# Monkey-patch so numpy<2.0 can unpickle .pkl files written under numpy>=2.0
# (e.g. precompute_features.py run in a different environment than
# training). Importing this module applies the patch as a side effect --
# it's standalone (only depends on pickle/numpy), so copy it into a
# notebook/script directly if you need the fix somewhere outside this repo.
#
# Background: numpy 2.0 renamed its internal `numpy.core` package to
# `numpy._core` (numpy.core survives only as a forward-compat alias in
# 2.0+). Every ndarray pickled under numpy>=2.0 embeds `numpy._core...`
# module paths for reconstruction. numpy<2.0 has no `numpy._core` package
# at all, so a plain `pickle.load()` on such a file raises
# `ModuleNotFoundError: No module named 'numpy._core'`. This patches
# pickle.load/pickle.loads (process-wide) to transparently rewrite any
# `numpy._core*` module reference to the equivalent, already-existing
# `numpy.core*` module before pickle resolves it -- a read-only rename
# that doesn't change what actually gets reconstructed.
#
# Scope: verified safe for plain numeric ndarrays (float32/int32/etc, no
# structured dtypes, object arrays, or masked arrays) -- exactly what
# precompute_features.py stores. More exotic numpy objects pickled under
# 2.0 may need a broader compat shim than this.
#
# No-op if numpy is already >=2.0 (nothing to patch).

import io
import pickle

import numpy as np


class _Numpy2CompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'numpy._core' or module.startswith('numpy._core.'):
            module = 'numpy.core' + module[len('numpy._core'):]
        return super().find_class(module, name)


def _numpy2_compat_load(file, **kwargs):
    return _Numpy2CompatUnpickler(file, **kwargs).load()


def _numpy2_compat_loads(data, **kwargs):
    return _Numpy2CompatUnpickler(io.BytesIO(data), **kwargs).load()


def patch() -> bool:
    """Apply the patch. Returns True if applied, False if numpy is already
    >=2.0 (nothing to do). Idempotent -- safe to call more than once."""
    if int(np.__version__.split('.')[0]) >= 2:
        return False
    pickle.load = _numpy2_compat_load
    pickle.loads = _numpy2_compat_loads
    return True


patch()
