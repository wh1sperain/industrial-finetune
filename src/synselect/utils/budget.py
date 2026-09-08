"""Wall-clock budgeting independent of torch; callers supply clocks for tests."""
from __future__ import annotations

import time


class WallBudget:
    def __init__(self, seconds=None, deadline=None, reserve=0, clock=time.monotonic,
                 wall_clock=time.time):
        self.clock = clock
        now = clock()
        ends = []
        if seconds is not None:
            if seconds <= 0:
                raise ValueError("Budget seconds must be positive")
            ends.append(now + seconds)
        if deadline is not None:
            ends.append(now + deadline - wall_clock())
        if reserve < 0:
            raise ValueError("Budget reserve must be nonnegative")
        self.end = min(ends) if ends else float("inf")
        self.reserve = reserve

    def remaining(self):
        return self.end - self.clock()

    def should_stop(self, next_work_seconds=0):
        return self.remaining() <= self.reserve + next_work_seconds
