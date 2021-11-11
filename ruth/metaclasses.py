"""Metaclasses module."""

from typing import Dict


class Singleton(type):
    """Metaclass for creating singleton objects."""

    _instances: Dict[int, object] = {}

    def __call__(cls, *args, **kwargs):
        try:
            args_ = map(lambda arg: arg if not isinstance(arg, dict) else frozenset(arg), args)
            kwargs_ = map(
                lambda kv: (kv[0], kv[1] if not isinstance(kv[1], dict) else frozenset(kv[1])),
                kwargs.items())

            h = hash((cls, tuple(args_), tuple(kwargs_)))
            if h not in cls._instances:
                cls._instances[h] = super(Singleton, cls).__call__(*args, **kwargs)
            return cls._instances[h]

        except TypeError:
            print(f"From the provided parameters cannot be created a hash. {cls}/{args}/{kwargs}")
            raise
