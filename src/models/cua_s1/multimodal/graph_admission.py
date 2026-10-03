"""Bounded request-frequency admission and sliding capture-work budget."""

from collections import OrderedDict, deque


class AdmissionPolicy:
    """Called under the runtime lock; counters advance once per predict request.

    A completed attempt can exceed the time budget because capture is synchronous.
    Every subsequent attempt waits for enough budget to expire. Cache hits never
    spend capture budget. History is intentionally bounded and forgetting is cold.
    """

    def __init__(self, config):
        self.config = config
        self.history = OrderedDict()
        self.cooldowns = OrderedDict()
        self.attempts = deque()
        self.request_index = 0

    def reset(self):
        self.history.clear()
        self.cooldowns.clear()
        self.attempts.clear()
        self.request_index = 0

    def begin_request(self):
        self.request_index += 1
        while self.attempts and (
            self.request_index - self.attempts[0][0] >= self.config.capture_window
        ):
            self.attempts.popleft()

    def reason(self, key):
        """Observe a cache miss and return its eager fallback reason, if any."""
        if self.request_index <= self.cooldowns.get(key, -1):
            return "cooldown"
        self.cooldowns.pop(key, None)
        last, count = self.history.get(key, (-self.config.admission_window, 0))
        if self.request_index - last > self.config.admission_window:
            count = 0
        if last != self.request_index:
            count += 1
        self.history[key] = (self.request_index, count)
        self.history.move_to_end(key)
        while len(self.history) > 128:
            self.history.popitem(last=False)
        if count < self.config.min_uses:
            return "warmup"
        if (
            len(self.attempts) >= self.config.max_captures
            or sum(item[1] for item in self.attempts) >= self.config.capture_budget_ms
        ):
            return "capture_budget"
        return None

    def start_capture(self):
        ticket = [self.request_index, 0.0]
        self.attempts.append(ticket)
        return ticket

    @staticmethod
    def finish_capture(ticket, elapsed_ms):
        ticket[1] = elapsed_ms

    def evict(self, key):
        self.history.pop(key, None)
        self.cooldowns[key] = self.request_index + self.config.cooldown_requests
        self.cooldowns.move_to_end(key)
        while len(self.cooldowns) > 128:
            self.cooldowns.popitem(last=False)
