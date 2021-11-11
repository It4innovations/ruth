"""A mapping between pandas DataFrame row and dataclass."""

import pandas as pd
from dataclasses import is_dataclass, asdict
from functools import wraps


class DataFrameRow:
    """Decorator for conversion between dataclass and pandas.Series"""

    def __init__(self, data_cls):
        assert is_dataclass(data_cls), "Expects a dataclass."
        self.data_cls = data_cls

    def __call__(self, fn):
        """The wrapped function must take as the first argument an instance of
        the type of `dataclass` and must return a `dataclass` object.
        """
        @wraps(fn)
        def wrapper(row, *args, **kwargs):
            obj = self._from_row(self.data_cls, row)
            result = fn(obj, *args, **kwargs)
            return self._to_row(result)
        return wrapper

    @staticmethod
    def _from_row(cls, row: pd.Series):
        """Converts the row into the dataclass."""
        return cls(**row.to_dict())

    @staticmethod
    def _to_row(data_object) -> pd.Series:
        """Converts the dataclass into a dataframe row."""
        return pd.Series(asdict(data_object))
