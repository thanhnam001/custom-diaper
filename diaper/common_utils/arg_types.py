#!/usr/bin/env python3

# Licensed under the MIT license.


def str2bool(v):
    """argparse/yamlargparse `type=` callable for boolean flags.

    `type=bool` is a classic argparse trap: argparse only calls `type` on
    strings coming from the CLI, and `bool("false")`/`bool("False")` are both
    `True` since any non-empty string is truthy -- so `--flag false` silently
    sets `flag=True`, indistinguishable from `--flag true`. YAML-sourced
    values are unaffected (PyYAML already parses `true`/`false` into real
    Python bools before the parser sees them), which is why this only bites
    CLI overrides of a config file's boolean settings.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError(f'Boolean value expected, got {v!r}.')
