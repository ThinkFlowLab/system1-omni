"""Bounded, causal mode selection with conservative capture amortization.

The 32-request observation window forecasts at most the next 64 requests. The
500 ms capture and 0.45/0.65 replay priors are heuristics for the pinned fallback
path, replaced by observations when available. No future workload is consulted.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass, field


@dataclass
class _Layout:
    bucket: tuple
    length: int
    visits: list = field(default_factory=list)
    mode: str | None = None
    switched_at: int = 0


class AutoSelector:
    window = 32
    max_layouts = 128
    switch_cooldown = 32
    margin = 1.25

    def __init__(self, config):
        self.config = config
        self.history = OrderedDict()
        self.costs = OrderedDict()
        self.index = 0
        self.last_reason = "cold"

    def reset(self):
        self.history.clear()
        self.costs.clear()
        self.index = 0

    def begin_request(self, index):
        self.index = index
        for key, layout in list(self.history.items()):
            layout.visits[:] = [
                visit for visit in layout.visits if index - visit[0] < self.window
            ]
            if not layout.visits:
                del self.history[key]

    def observe(self, key, bucket, length):
        layout = self.history.setdefault(key, _Layout(bucket, length))
        if layout.visits and layout.visits[-1][0] == self.index:
            layout.visits[-1][1] += 1
        else:
            layout.visits.append([self.index, 1])
        self.history.move_to_end(key)
        while len(self.history) > self.max_layouts:
            self.history.popitem(last=False)

    def record(
        self, bucket, mode, *, latency_ms=None, capture_ms=None, owned_bytes=None
    ):
        costs = self.costs.setdefault(bucket, {})
        for key, value in [(mode, latency_ms), (mode + "_capture", capture_ms)]:
            if value is not None and math.isfinite(value) and value > 0:
                costs[key] = (
                    value if key not in costs else 0.8 * costs[key] + 0.2 * value
                )
        if owned_bytes is not None and owned_bytes > 0:
            costs[mode + "_bytes"] = max(costs.get(mode + "_bytes", 0), owned_bytes)
        self.costs.move_to_end(bucket)
        while len(self.costs) > self.max_layouts:
            self.costs.popitem(last=False)

    def selected(self, key, mode):
        layout = self.history.get(key)
        if layout is not None and layout.mode != mode:
            layout.mode = mode
            layout.switched_at = self.index

    def _result(self, mode, reason):
        self.last_reason = reason
        return mode

    def choose(
        self,
        key,
        bucket,
        length,
        *,
        exact_entry=None,
        bucket_entry=None,
        cache_shapes=0,
        cache_bytes=0,
        exact_allowed=True,
        bucket_allowed=True,
    ):
        costs = self.costs.get(bucket, {})
        eager = costs.get("eager", 100.0)
        exact = costs.get("exact", eager * 0.45)
        rule = costs.get("rule-bucket", eager * 0.65)
        if exact_entry is not None:
            return self._result("exact", "resident_exact")
        layout = self.history[key]
        required = max(2, self.config.min_uses)
        if bucket_entry is not None and length in bucket_entry.rejected_lengths:
            bucket_allowed = False
        if len(layout.visits) < required:
            # A previously verified resident bucket length needs no extra gate.
            if (
                bucket_allowed
                and bucket_entry is not None
                and length in bucket_entry.verified_lengths
                and rule < eager
            ):
                return self._result("rule-bucket", "resident_bucket")
            return self._result("eager", "cold")
        gap = (layout.visits[-1][0] - layout.visits[0][0]) / (len(layout.visits) - 1)
        copies = sum(v[1] for v in layout.visits) / len(layout.visits)
        future_exact = 2 * self.window / gap * copies
        repeated = sum(len(item.visits) >= required for item in self.history.values())
        pressure = repeated > max(1, self.config.max_shapes // 2)
        exact_fits = (
            cache_shapes < self.config.max_shapes
            and cache_bytes + costs.get("exact_bytes", 0) <= self.config.max_bytes
        )
        switching = layout.mode not in (None, "exact")
        cooled = (
            not switching or self.index - layout.switched_at >= self.switch_cooldown
        )
        current_cost = (
            min(eager, rule) if bucket_allowed and bucket_entry is not None else eager
        )
        if (
            exact_allowed
            and gap <= self.config.admission_window
            and not pressure
            and exact_fits
            and cooled
            and future_exact * max(0, current_cost - exact)
            > self.margin * costs.get("exact_capture", 500.0)
        ):
            return self._result("exact", "amortized_exact")

        if not bucket_allowed:
            return self._result("eager", "bucket_rejected")
        if bucket_entry is not None:
            gate = 0 if length in bucket_entry.verified_lengths else eager
            if future_exact * max(0, eager - rule) > self.margin * gate:
                return self._result("rule-bucket", "resident_bucket")
            return self._result("eager", "length_gate_cost")
        if (
            layout.mode == "exact"
            and self.index - layout.switched_at < self.switch_cooldown
        ):
            return self._result("eager", "switch_cooldown")

        visits = {}
        lengths = set()
        for item in self.history.values():
            if item.bucket == bucket:
                lengths.add(item.length)
                for request, count in item.visits:
                    visits[request] = visits.get(request, 0) + count
        span = max(visits) - min(visits)
        future_bucket = (
            2
            * self.window
            * (len(visits) - 1)
            / span
            * sum(visits.values())
            / len(visits)
        )
        # The capture estimate already includes the current length's reference.
        gates = max(0, len(lengths) - 1) * eager
        if (
            future_bucket * max(0, eager - rule)
            > self.margin * costs.get("rule-bucket_capture", 500.0) + gates
        ):
            return self._result("rule-bucket", "amortized_bucket")
        return self._result("eager", "cost")
