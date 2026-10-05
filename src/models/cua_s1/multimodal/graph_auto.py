"""Opt-in adaptive dispatch with shared Graph budgets and strict numeric gates."""

from collections import Counter, deque
from contextlib import contextmanager

from .graph_buckets import bucket_length
from .graph_runtime import GraphConfig
from .graph_selector import AutoSelector
from .graph_shared import SharedGraphRuntime


class AutoGraphRuntime(SharedGraphRuntime):
    def __init__(self, model, config=None):
        super().__init__(model, config or GraphConfig(mode="auto"))
        self.selector = AutoSelector(self.config)
        self._pending = deque(maxlen=16)
        self._decisions = {}
        self.decisions = Counter()
        self.selections = Counter(eager=0, exact=0, **{"rule-bucket": 0})

    @property
    def stats(self):
        result = super().stats
        result.update(
            {
                "selected_" + mode.replace("-", "_"): count
                for mode, count in self.selections.items()
            }
        )
        return result

    def select_mode(self, mode):
        raise RuntimeError("automatic runtime chooses its own execution mode")

    def _collect_timings(self):
        while self._pending and self._pending[0][1].query():
            start, end, bucket, mode = self._pending.popleft()
            self.selector.record(bucket, mode, latency_ms=start.elapsed_time(end))

    @contextmanager
    def request(self):
        with super().request():
            self._collect_timings()
            self.selector.begin_request(self.admission.request_index)
            self._decisions.clear()
            yield

    def forward(self, values):
        import torch

        with self.lock, torch.no_grad():
            if self._closed:
                raise RuntimeError("shared Graph runtime is closed")
            exact = self._runtimes["exact"]
            bucket_runtime = self._runtimes["rule-bucket"]
            if not self._in_request:
                exact.stats["no_request"] += 1
                return exact._eager(values)
            if (
                not exact._supported(values)
                or bucket_length(
                    values["inputs_embeds"].shape[1], self.config.bucket_width
                )
                > self.config.max_tokens
            ):
                exact.stats["unsupported"] += 1
                self.selections["eager"] += 1
                return exact._eager(values)
            key, bucket = exact._key(values), bucket_runtime._key(values)
            if (
                key not in self._decisions
                and len(self._decisions) >= self.selector.max_layouts
            ):
                self.selections["eager"] += 1
                self.decisions["request_layout_limit"] += 1
                return exact._eager(values)
            length = values["inputs_embeds"].shape[1]
            self._collect_timings()
            self.selector.observe(key, bucket, length)
            # Observe both candidate histories once/request, including eager
            # observations. reason() deduplicates another call by the selected
            # runtime; discarded answers do not spend capture budget.
            exact.admission.reason(key)
            bucket_runtime.admission.reason(bucket)
            if key not in self._decisions:
                mode = self.selector.choose(
                    key,
                    bucket,
                    length,
                    exact_entry=self.cache.entries.get(("exact", key)),
                    bucket_entry=self.cache.entries.get(("rule-bucket", bucket)),
                    cache_shapes=len(self.cache),
                    cache_bytes=self.cache.bytes,
                    exact_allowed=key not in exact.disabled,
                    bucket_allowed=bucket not in bucket_runtime.disabled,
                )
                self._decisions[key] = mode
                self.decisions[self.selector.last_reason] += 1
            mode = self._decisions[key]
            self.selections[mode] += 1
            runtime = exact if mode == "eager" else self._runtimes[mode]
            before = dict(runtime.stats)
            samples = sum(self.selections.values())
            # Keep overhead bounded even when a selected mode repeatedly falls
            # back and therefore never acquires its own replay-cost sample.
            sample = samples <= 8 or samples % 16 == 0
            start = end = None
            with torch.cuda.device(values["inputs_embeds"].device):
                if sample:
                    start, end = (
                        torch.cuda.Event(enable_timing=True) for _ in range(2)
                    )
                    start.record()
                result = (
                    runtime._eager(values)
                    if mode == "eager"
                    else runtime.forward(values)
                )
                if end is not None:
                    end.record()
            attempts = runtime.stats["capture_attempts"] - before["capture_attempts"]
            checks = runtime.stats.get("length_checks", 0) - before.get(
                "length_checks", 0
            )
            if attempts:
                self.selector.record(
                    bucket,
                    mode,
                    capture_ms=runtime.stats["capture_attempt_ms"]
                    - before["capture_attempt_ms"],
                )
            elif not checks and start is not None:
                actual_mode = (
                    mode if runtime.stats["replays"] > before["replays"] else "eager"
                )
                self._pending.append((start, end, bucket, actual_mode))
            entry_key = key if mode == "exact" else bucket
            entry = self.cache.entries.get((mode, entry_key))
            if entry is not None:
                self.selector.record(bucket, mode, owned_bytes=entry.bytes)
                self.selector.selected(key, mode)
            return result

    def invalidate(self):
        with self.lock:
            super().invalidate()
            self.selector.reset()
            self._pending.clear()
            self._decisions.clear()
