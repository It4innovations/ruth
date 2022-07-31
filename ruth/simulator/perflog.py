
import math
from dataclasses import dataclass
from datetime import timedelta

@dataclass
class StepInfo:
    step: int
    n_active: int
    gv_size: int
    time_for_alternatives: timedelta
    time_for_ptdr: timedelta
    time_for_advance: timedelta
    time_for_gv_shift: timedelta

    @staticmethod
    def from_row(step, n_active, gv_size,
                 time_for_alternatives_s, time_for_ptdr_s, time_for_advance_s, time_for_gv_shift_s):
        time_for_ptdr = None if math.isnan(time_for_ptdr_s) else timedelta(seconds=time_for_ptdr_s)
        time_for_advance = None if math.isnan(time_for_advance_s) else timedelta(seconds=time_for_advance_s)
        time_for_gv_shift = None if math.isnan(time_for_gv_shift_s) else timedelta(seconds=time_for_gv_shift_s)

        return StepInfo(step, n_active, gv_size,
                        timedelta(seconds=time_for_alternatives_s),
                        time_for_ptdr,
                        time_for_advance,
                        time_for_gv_shift)

    def __repr__(self):
        sec = timedelta(seconds=1)
        time_for_ptdr = "" if self.time_for_ptdr is None else f"{self.time_for_ptdr / sec}"
        time_for_advance = "" if self.time_for_advance is None else f"{self.time_for_advance / sec}"
        time_for_gv_shift = "" if self.time_for_gv_shift is None else f"{self.time_for_gv_shift / sec}"
        return f"{self.step};{self.n_active};{self.gv_size};{self.time_for_alternatives / sec};{time_for_ptdr};" \
               f"{time_for_advance};{time_for_gv_shift}"

    def __str__(self):
        sec = timedelta(seconds=1)
        return f"StepInfo(step={self.step}, active={self.n_active}, gv_size={self.gv_size})"