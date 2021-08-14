"""Metaclasses module."""

from typing import Dict


class Singleton(type):
    """Metaclass for creating singleton objects."""

    _instances: Dict[type, object] = {}

    def __call__(cls, *args, **kwargs):
        """Create a new singleton or use the one instatiated previously."""
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
