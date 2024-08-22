from enum import Enum
from datetime import timedelta


class TimeUnit(Enum):
    SECONDS = timedelta(seconds=1)
    MINUTES = timedelta(minutes=1)
    HOURS = timedelta(hours=1)

    @staticmethod
    def from_str(name):
        if name == 'seconds':
            return TimeUnit.SECONDS
        elif name == 'minutes':
            return TimeUnit.MINUTES
        elif name == 'hours':
            return TimeUnit.HOURS

        raise Exception(f"Invalid time unit: '{name}'.")
