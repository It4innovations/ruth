"""A module for uninifying different sources of level of service."""

from datetime import timedelta
from probduration import Segment, HistoryHandler
from .metaclasses import Singleton

class LoSDb:

    def get(self, datetime, segment: Segment, toss: float) -> float:
        """Provides a level of service for the given segment at datetime.

        Parameters:
        -----------
        datetime: datetime.datetime
            A datetime at which the level of service is interested.
        segment: Segmetn
            A segment at which the level of service is interested.
        toss: float (0.0..1.0)
            A "dice toss" based on which a particular LoS is chosen
        from the probability profile.
        """
        pass


class ProbProfileDb(LoSDb):

    def __init__(self, prob_profiles):
        self.prob_profiles = prob_profiles

    def get(self, datatime, segment, toss):
        return self.prob_profiles.level_of_service(datetime, segment.id, toss)


class FreeFlowDb(LoSDb, metaclass=Singleton):

    def __init__(self):
        self.ff_profiles = HistoryHandler.no_limit()

    def get(self, datetime, segment, toss):
        return self.ff_profiles.level_of_service(datetime, segment.id, toss)


class GlobalViewDb(LoSDb):

    def __init__(self, gv):
        self.gv = gv

    def get(self, datetime, segment, toss_=None):
        """Provides the level of service for the given segment at given datetime.

        The version with global view ignores the `toss` as there is no randomness.

        Returns a level of service at segment in time +- 5s.
        """
        return self.gv.level_of_service_in_time_at_segment(datetime, segment, timedelta(seconds=5))
