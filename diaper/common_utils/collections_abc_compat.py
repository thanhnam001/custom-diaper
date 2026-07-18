#!/usr/bin/env python3

# Monkey-patch so pre-3.10-era libraries (old torch/transformers/etc, like
# this repo's pinned torch==1.10.0 and the fnlandini transformers fork)
# that still reach for `collections.Container`/`collections.Mapping`/etc
# keep working on Python 3.10+.
#
# Background: Container, Mapping, MutableMapping, Sequence,
# MutableSequence, Iterable, Iterator, Callable, Hashable, Sized, etc. were
# ABCs that lived directly under `collections` up through Python 3.9, as a
# deprecated alias for their real home, `collections.abc` (which has
# existed since Python 3.3). Python 3.10 removed the deprecated top-level
# aliases entirely, so any code still doing `collections.Container` (rather
# than `collections.abc.Container`) raises
# `AttributeError: module 'collections' has no attribute 'Container'`.
# Libraries pinned to old versions predate that removal and were never
# updated for it.
#
# Importing this module re-adds those names onto the `collections` module
# as aliases of the real `collections.abc` classes -- exactly what they
# always resolved to before the deprecated alias was removed, so this
# doesn't change behavior for anything, it just restores the old (still
# perfectly valid) access path.
#
# No-op on Python <3.10 (nothing removed there).

import collections
import collections.abc
import sys

_ABC_NAMES = [
    'Awaitable', 'Coroutine', 'AsyncIterable', 'AsyncIterator',
    'AsyncGenerator', 'Hashable', 'Iterable', 'Iterator', 'Generator',
    'Reversible', 'Sized', 'Container', 'Callable', 'Collection', 'Set',
    'MutableSet', 'Mapping', 'MutableMapping', 'MappingView', 'KeysView',
    'ItemsView', 'ValuesView', 'Sequence', 'MutableSequence', 'ByteString',
]


def patch() -> bool:
    """Apply the patch. Returns True if any alias was (re-)added, False if
    nothing needed patching. Idempotent -- safe to call more than once."""
    if sys.version_info < (3, 10):
        return False
    applied = False
    for name in _ABC_NAMES:
        if not hasattr(collections, name) and hasattr(collections.abc, name):
            setattr(collections, name, getattr(collections.abc, name))
            applied = True
    return applied


patch()
